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
  runs runnable alone (the walk to the place takes the facing its run
  wants)."""
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
  assert set(library.runnable()) == {"lay_cup", "lay_plate", "lay_fork"}


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


def test_a_reference_source_with_no_def_is_refused_by_name():
  """`job_routine` defines each source under its first def's name: one with
  none is a ValueError saying so (the second review of #682's stage C) --
  never a StopIteration, which inside the routine is an opaque RuntimeError."""
  import pytest
  from pluggybot.challenge import solutions
  with pytest.raises(ValueError, match="no def"):
    solutions._def_name("x = 1\n")
  assert solutions._def_name(solutions.PASS[0]) == "lay_cup"


def _calls(source: str) -> list[tuple[str, list]]:
  """A source's calls in the order they are written: each verb's name and
  its arguments, a constant as its value and anything else as its text."""
  out = []
  for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
      args = [a.value if isinstance(a, ast.Constant) else ast.unparse(a) for a in node.args]
      out.append(((node.lineno, node.col_offset), node.func.id, args))
  return [(name, args) for _, name, args in sorted(out)]


def _index(calls: list, test, what: str, start: int = 0) -> int:
  """The first of `calls` from `start` that passes `test` -- a source
  without one fails saying which, never a StopIteration."""
  i = next((j for j in range(start, len(calls)) if test(calls[j])), None)
  assert i is not None, f"the reference's source has no {what}"
  return i


def test_the_pass_plate_is_eased_before_it_is_let_go():
  """#682's stage C, round 3: the plate tilts in its jaws as it is carried
  (2-28 deg at its put-down's hover), and the descent's touch flattens most
  of them; one still tilted 10-24 deg was let go 1.5-3.7 cm over the table
  and came down at 0.5-0.63 m/s, a drop. Eased to the gripper's least
  squeeze first, the same flights settled flat in the jaws and were let go
  at the table's top (SimNotes, "The place laid from the pass"). Off the
  reference's own source: the plate's let-go at the place follows a shut
  at the least squeeze."""
  from pluggybot.challenge import solutions
  from pluggybot.procedure import steps
  calls = _calls(solutions.PROCEDURES["set_table_pass"][1])
  after = calls[_index(calls, lambda c: c[0] == "to_place", "walk to the place"):]
  let_gos = [i for i, (n, a) in enumerate(after) if n == "open_hand" and a == ["left"]]
  assert let_gos, "the plate is never let go at the place"
  before = [c for c in after[:let_gos[-1]] if c[0] in steps.HANDS_VERBS and c[1][:1] == ["left"]]
  last = before[-1][0] if before else "nothing"
  assert last == "close_hand", f"the plate is let go straight after {last}"
  assert before[-1][1][1] <= steps._GRIP.lo + 1e-9, f"eased to {before[-1][1][1]} N, not the least squeeze"


def _hand_moves(source: str, hand: str) -> list[tuple[str, list]]:
  """The verbs and procedures a run calls on `hand`, in order: a hand verb
  whose first argument is the hand, or `go` (the reference's own, a move
  of the hand it is passed)."""
  from pluggybot.procedure import steps
  return [(n, a) for n, a in _calls(source)
          if (n in steps.HANDS_VERBS or n == "go") and a[:1] == [hand]]


def test_the_pass_plate_is_brought_back_from_the_pass_by_a_position_move():
  """#682's stage C, round 3: the plate gripped at the pass is brought back
  over the counter's edge before the walk -- far enough that the walk's
  hand-over does not strike the counter with it (from the table check's
  stance it struck in 30 of 32 legs and tilted in its jaws) -- and brought
  back by `pull` (a force move, `go`'s) further than about 0.22 m the robot
  fell: pulled back 0.26-0.30 m, 12 of 50, its posture taken again as the
  arm missed (a 20 cm crouch, a 31 deg bow over the counter and a 15 deg
  waist turn, traced). By `move_hand`, one posture for the whole path (a
  crouch, no bow), the plate taken from 0.50 m and brought back to 0.23
  stood in 32 of 32 and laid 23 where the table check's stance laid 17
  (SimNotes, "The place laid from the pass"); since round 5 in two such
  moves (the next test). Off the reference's own source: what the plate's
  hand does after the lift that takes it off the pass is a position move."""
  from pluggybot.challenge import solutions
  moves = _hand_moves(solutions.PROCEDURES["set_table_pass"][1], "left")
  shut = _index(moves, lambda m: m[0] == "close_hand", "shut on the plate at the pass")
  lift = _index(moves, lambda m: m[0] == "pull", "lift off the pass", start=shut + 1)
  back = moves[lift + 1][0] if lift + 1 < len(moves) else "nothing"
  assert back == "move_hand", f"the plate is brought back by {back}"


def _plate_pull_back(monkeypatch, gx: float, gy: float, gz: float = 1.0) -> list:
  """The pass reference's plate pull-back off the pass as the language runs
  it for a grip at (gx, gy, gz) in the stance frame: its run compiled as the
  library compiles it, and its statements from the lift off the pass to the
  walk to the place worked out by the language's own evaluator
  (`lang._eval`), the hand read where its moves left it; every `move_hand`
  of the left hand is recorded where it was asked to go, taken as arrived.
  A statement the walk does not know fails, never one followed otherwise."""
  from types import SimpleNamespace
  from pluggybot.challenge import solutions
  from pluggybot.lifecycle import HUMANOID_HOME, world_facts
  from pluggybot.procedure import axes, lang
  from pluggybot.procedure import library as lib
  facts = world_facts(HUMANOID_HOME)
  library = lib.Library(facts)
  for source in (*solutions.LIBRARY["set_table_pass"], solutions.PROCEDURES["set_table_pass"][1]):
    library.define(solutions._def_name(source), source)
  hand = {"left.x": gx, "left.y": gy, "left.z": gz}
  for k in hand:
    monkeypatch.setitem(axes.SENSORS, k, SimpleNamespace(read=lambda life, k=k: hand[k]))
  env: dict = {}
  moves = []

  def value(expr):
    return lang._eval(expr, env, SimpleNamespace(), facts)

  def moved(verb, args):
    if verb == "move_hand" and args["hand"] == ("str", "left"):
      x, y, z = (value(args[a]) for a in ("x", "y", "z"))
      moves.append((x, y, z))
      hand.update({"left.x": x, "left.y": y, "left.z": z})

  def run(stmts) -> bool:
    """True once the walk to the place is reached."""
    for s in stmts:
      if s[0] == "verb":
        moved(s[1], s[2])
      elif s[0] == "set":
        env[s[1]] = value(s[2])
      elif s[0] == "if":
        for cond, body in s[1]:
          if cond[0] == "try":
            assert cond[2][0] == "verb", f"the pull-back tests a {cond[2][0]}"
            moved(cond[2][1], cond[2][2])
            held = not cond[1]  # the verb did what it said
          else:
            held = lang._truth(value(cond))
          if held:
            if run(body):
              return True
            break
        else:
          if run(s[2]):
            return True
      elif s[0] == "call" and s[1] == "to_place":
        return True
      elif s[0] != "pass":
        raise AssertionError(f"the plate's pull-back holds a {s[0]} this walk does not follow")
    return False
  body = library.entries["lay_plate"].procedure.body
  lift = next(i for i, s in enumerate(body) if s[0] == "verb" and s[1] == "pull"
              and s[2]["hand"] == ("str", "left") and s[2]["dz"] == ("num", 1.0))
  assert run(body[lift + 1:]), "the plate's run never walks to the place"
  return moves


def test_the_pass_plate_is_brought_back_in_two_moves_ending_out_of_the_middle(monkeypatch):
  """#682's stage C, round 5: the plate gripped at the pass, 0.42-0.51 m
  out, was brought back to 0.23 m by one position move along the stance's
  x -- and a grip the hand read 0.20 m left of the stance or nearer its
  middle (`near` lands it 0.12-0.32 m out) took a posture with the waist
  turned 30-60 deg for that path: it held back, tipped the plate 19-44 deg
  in its jaws (then let go on the walk) or fell, the pelvis coming
  0.25-0.29 m toward the pass (round 4's kitchen_22; on round 5's tree 3 of
  64 plate legs on dev draws, and with the grip set 6 cm nearer the middle
  on purpose 7 falls, 29 plates dropped and 27 laid of 64). In two moves,
  halfway first, the last ending no nearer the middle than
  `PASS_PLATE_LEFT`: 52 of those 64 laid, no fall, 8 dropped (SimNotes,
  "The place laid from the pass"). Off the reference's own source, run by
  the language for grips round where it grips."""
  from pluggybot.challenge import solutions
  for gx in (0.43, 0.47, 0.51):
    for gy in (0.12, 0.2, 0.3, 0.36):
      moves = _plate_pull_back(monkeypatch, gx, gy)
      assert len(moves) == 2, f"a grip at ({gx}, {gy}) is brought back by {len(moves)} moves"
      (x1, y1, _), (x2, y2, _) = moves
      assert abs(x1 - (gx + x2) / 2) < 1e-9 and y1 == gy, (gx, gy, moves)
      assert x2 == solutions.PASS_PLATE_BACK and y2 == max(gy, solutions.PASS_PLATE_LEFT), \
        (gx, gy, moves)


def test_the_pass_cup_descends_to_its_grip_by_a_position_move():
  """#682's stage C, round 3: the cup is reached for 10 cm over its grip and
  then brought down to it. Brought down by a `pull`, its grip at the shut
  drifted 1.6 cm across the jaws (sd), and a pad landed on the lid's rim
  (2-5 of 32: the jaws clear the cup by 1 cm a side); reached for 5 cm over
  it, the arm took another configuration at the grip and its pull-back off
  the pass ran out of reach 14 of 30. By `move_hand` from the 10 cm hover
  the arm's configuration comes down with it and the grip lands within 0.2
  cm across (sd), none on the rim (SimNotes, "The place laid from the
  pass"). Off the reference's own source: the cup's hand goes from its
  reach to its shut by a position move."""
  from pluggybot.challenge import solutions
  moves = _hand_moves(solutions.PROCEDURES["set_table_pass"][0], "right")
  reach = _index(moves, lambda m: m[0] == "reach", "reach for the cup")
  shut = _index(moves, lambda m: m[0] == "close_hand", "shut on the cup", start=reach + 1)
  assert moves[shut - 1][0] == "move_hand", f"the cup is brought down to its grip by {moves[shut - 1][0]}"


def test_the_pass_cup_pick_is_tried_again_where_its_descent_is_held_back():
  """#682's stage C, round 4: the cup is brought down to its grip from its
  10 cm hover by a position move. Where the hover landed 0.8 cm or more off
  the cup's middle the jaws came down on its lid and the move read held
  back -- and in every one of round 3's 26 such picks the pick failed (the
  cup tipped 34-46 deg in the jaws, the lift read it let go, or the shut
  closed on nothing). Tried again from the stance -- the arm home, posed and
  the cup looked at again -- a fresh hover lands within the jaws' 1 cm a
  side: 7 of the 9 held back in the bar's fourth reading were gripped so
  (SimNotes, "The place laid from the pass"). Off the reference's own
  source: the descent's test, read false, goes home, looks at the cup
  again, reaches for its hover and comes down again to the same height,
  in that order, before the shut."""
  from pluggybot.challenge import solutions
  tree = ast.parse(solutions.PROCEDURES["set_table_pass"][0])
  tests = sorted((n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.UnaryOp) and isinstance(n.test.operand, ast.Call)
                  and getattr(n.test.operand.func, "id", "") == "move_hand"), key=lambda n: n.lineno)
  assert tests, "the cup's descent to its grip is not a test"
  descent = [ast.unparse(a) for a in tests[0].test.operand.args]
  body = _calls(ast.unparse(ast.Module(body=tests[0].body, type_ignores=[])))
  verbs = [(n, a) for n, a in body if n in ("withdraw", "spot", "reach", "move_hand")]
  home = _index(verbs, lambda v: v[0] == "withdraw", "way home, held back")
  look = _index(verbs, lambda v: v[0] == "spot", "look at the cup again", start=home + 1)
  hover = _index(verbs, lambda v: v[0] == "reach", "reach for the hover again", start=look + 1)
  again = _index(verbs, lambda v: v[0] == "move_hand", "descent again", start=hover + 1)
  assert verbs[again][1][1:4] == descent[1:4], (
    f"the second descent goes to {verbs[again][1][1:4]}, the first to {descent[1:4]}")

def test_the_pass_fork_is_brought_back_from_the_pass_in_two_pulls():
  """#682's stage C, round 4: the fork gripped at the pass is pulled back
  over the counter's edge to 0.21 m before the walk. Pulled back 0.24 m or
  more in one pull -- a grip 0.45 m out or further -- the pull took a 31
  deg bow and a 20 cm crouch for its way, stood up out of it within 0.3 s
  as its arm missed, and the robot fell in 6 of 21 (a shorter pull in none
  of 79): every fall of round 3's reading (5 of 64) and of round 4's base on
  dev draws. In two pulls, the first
  stopping at `PASS_FORK_MID`, each asks a posture that reaches both its
  ends standing (SimNotes, "The place laid from the pass"). Off the
  reference's own source: after the lift off the pass, the fork's hand is
  brought back by two moves, the first stopping short of the second."""
  from pluggybot.challenge import solutions
  moves = _hand_moves(solutions.PROCEDURES["set_table_pass"][2], "left")
  shut = _index(moves, lambda m: m[0] == "close_hand", "shut on the fork at the pass")
  lift = _index(moves, lambda m: m[0] == "pull", "lift off the pass", start=shut + 1)
  back = moves[lift + 1:lift + 3]
  assert [m[0] for m in back] == ["go", "go"], f"the fork is brought back by {back}"
  first, last = (float(m[1][1]) for m in back)
  assert first == solutions.PASS_FORK_MID and first - last >= 0.1, (first, last)
