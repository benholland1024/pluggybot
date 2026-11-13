"""Ladder A of issue #264: what a hand-written solution stands on -- the
humanoids' hand jobs' (#559: `challenge/solutions.py`, flown by
`scripts/solve.py --feature drawer|door|cube|chest|table`).

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
  from pluggybot.lifecycle import HUMANOID_HOME, task_producer, world_targets
  board = TaskBoard(path=str(tmp_path / "t.json"))
  producer = task_producer(board, HUMANOID_HOME, procedures=True)
  offered = list(producer.cadence.kinds)
  assert offered
  here = world_targets(HUMANOID_HOME, procedures=True)
  for kind in offered:
    area = here[KINDS[kind].target_kind][0]
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


def test_the_pass_reference_is_a_library_the_served_language_takes():
  """The place laid from the pass (#682's stage C): the procedures its runs
  call and its three runs, a piece each, compile against the served pair's
  language (calls, the hands' verbs, the maths) in one library -- each
  source inside the language's cap, the whole inside a library's -- the
  procedures they call defined first (`solutions.LIBRARY`), and only the
  runs and the walk to the place runnable alone."""
  from pluggybot.challenge import solutions
  from pluggybot.lifecycle import HUMANOID_HOME, world_facts
  from pluggybot.procedure import lang
  from pluggybot.procedure import library as lib
  library = lib.Library(world_facts(HUMANOID_HOME))
  helpers = solutions.LIBRARY["set_table_pass"]
  runs = solutions.PROCEDURES["set_table_pass"]
  for source in (*helpers, *runs):
    assert len(source) <= lang.MAX_SOURCE_CHARS
    library.define(solutions._def_name(source), source)
  assert [solutions._def_name(s) for s in runs] == ["lay_cup", "lay_plate", "lay_fork"]
  assert all(e.valid for e in library.entries.values())
  assert set(library.runnable()) == {"to_place", "lay_cup", "lay_plate", "lay_fork"}


def _pass_cup_grip():
  """The pass reference's cup grip as its source says it: how far short of
  the cup's middle it is aimed along the fingers' way, how far under the
  lid, and the fingers' tilt down from level, rad."""
  import math
  from pluggybot.challenge import solutions
  tree = ast.parse(solutions.PROCEDURES["set_table_pass"][0])
  said = {}
  for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
      v, off = node.value, 0.0
      if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Sub) and isinstance(v.right, ast.Constant):
        v, off = v.left, v.right.value
      if (isinstance(v, ast.Call) and getattr(v.func, "id", "") == "read"
          and isinstance(v.args[0], ast.Constant)):
        said.setdefault(v.args[0].value, off)
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "reach" and "f" not in said:
      fx, _, fz = (ast.literal_eval(a) for a in node.args[4:7])
      said["f"] = math.atan2(-fz, fx)
  return said["spot.x"], said["spot.z"], said["f"]


def test_the_pass_cup_is_gripped_across_each_pads_width_off_its_tips():
  """#682's open item 2, the cup's creep: a cup gripped from above by
  fingers tilted down touches each pad along the line of its side, and
  that line crosses the pad's face where the grip is aimed along the
  fingers' way. Aimed 2.5 cm short of the cup's middle (b531491), the line
  met each pad only at its fingertip's corner -- one contact a pad (a
  contact trace: grip frame z -0.035..-0.040, the tips at -0.040), a pivot
  about the jaws' line -- and carried, the cup crept out of the jaws in 20
  of 32 flights on each walker; aimed at its middle, the line crosses the
  pad's width away from the tips (the trace's two contacts a pad: z -0.023
  and +0.007) and none of 64 let go. Off the reference's own source, the
  gripper's pads and the cup's drawing."""
  import math
  from pluggybot.activity import dishware as dw
  from pluggybot.humanoid import parts
  short, under, tilt = _pass_cup_grip()
  wy, wz = parts.GRIPPER.pad_half[1:]          # across the pad, along the finger
  c, s = math.cos(tilt), math.sin(tilt)
  h = dw.CUP_H - under                         # the grip over the cup's foot
  # the cup's foot in the grip frame (z back up the fingers, y across them),
  # its side's line on the pad's face, t up the cup's axis
  y0, z0 = h * c - short * s, -(short * c + h * s)
  on = [t / 1000 for t in range(int(dw.CUP_H * 1000) + 1)
        if abs(y0 - t / 1000 * c) <= wy and abs(z0 + t / 1000 * s) <= wz]
  assert on, "the cup's side never meets the pads"
  zs = [z0 + t * s for t in on]
  assert min(zs) + wz >= 0.01, f"its side meets each pad {100 * (min(zs) + wz):.1f} cm from the fingertips"
  assert on[-1] - on[0] >= 0.02, f"its side lies along {100 * (on[-1] - on[0]):.1f} cm of each pad"
