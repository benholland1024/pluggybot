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
import math
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
  # rule): the producer is handed the places and what a draw's sentence
  # names (#736: the shelf's labels, off the draw alone), and nothing else
  assert set(producer.facts) == {"places", "words"}
  from pluggybot.lifecycle import world_config
  assert all(producer.facts["words"](t, 3) == {} for t in world_config(HUMANOID_HOME)["jobs"])


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


def test_solve_refuses_a_feature_whose_kind_has_no_reference(monkeypatch):
  """`scripts/solve.py` refuses a feature whose kind has no hand-written
  reference (`refused`; #682's stage C, the full task, its review): unrefused,
  the job was offered, set out and claimed, and its run raised KeyError at
  `PROCEDURES` -- and it flies every feature whose kind has one."""
  from pathlib import Path
  from pluggybot.challenge import solutions
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
  import solve
  for feature, (kind, _) in solve.HAND_JOBS.items():
    assert (solve.refused(feature) is None) == (kind in solutions.PROCEDURES), feature
  monkeypatch.delitem(solutions.PROCEDURES, "set_table_pass")
  assert "no reference is written for set_table_pass" in solve.refused("pass")


def test_solve_names_every_start_one_flight_may_take(monkeypatch, capsys):
  """`--view` and `--strip` fly one flight, so `--from tour` is refused -- and
  the refusal names every start one flight may take, the pass's and the
  kitchen shelf's among them (#682's stage C, the full task, its review: it
  said "station or store", two of four)."""
  import sys
  from pathlib import Path
  import pytest
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
  import solve
  monkeypatch.setattr(sys, "argv", ["solve.py", "--feature", "kitchen", "--from", "tour", "--strip"])
  with pytest.raises(SystemExit):
    solve.main()
  said = capsys.readouterr().err
  for start in ("station", "store", "pass", "shelf"):
    assert start in said.split("fly one:")[1], said


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


def _plate_run():
  """The pass reference's plate run compiled as the library compiles it
  (its callees defined first), and the world's facts."""
  from pluggybot.challenge import solutions
  from pluggybot.lifecycle import HUMANOID_HOME, world_facts
  from pluggybot.procedure import library as lib
  facts = world_facts(HUMANOID_HOME)
  library = lib.Library(facts)
  for source in (*solutions.LIBRARY["set_table_pass"], solutions.PROCEDURES["set_table_pass"][1]):
    library.define(solutions._def_name(source), source)
  return library.entries["lay_plate"].procedure.body, facts


def _plate_pull_back(monkeypatch, run, gx: float, gy: float, gz: float = 1.0) -> list:
  """The plate's pull-back off the pass as the language runs it (`run`,
  `_plate_run`'s) for a grip at (gx, gy, gz) in the stance frame: its
  statements from the lift off the pass to the walk to the place worked out
  by the language's own evaluator (`lang._eval`), the hand read where its
  moves left it; every `move_hand` of the left hand is recorded where it was
  asked to go, taken as arrived. A statement the walk does not know fails,
  never one followed otherwise."""
  from types import SimpleNamespace
  from pluggybot.procedure import axes, lang
  body, facts = run
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
  run = _plate_run()
  for gx in (0.43, 0.47, 0.51):
    for gy in (0.08, 0.12, 0.2, solutions.PASS_PLATE_LEFT, 0.3, 0.36):
      moves = _plate_pull_back(monkeypatch, run, gx, gy)
      assert len(moves) == 2, f"a grip at ({gx}, {gy}) is brought back by {len(moves)} moves"
      (x1, y1, _), (x2, y2, _) = moves
      assert abs(x1 - (gx + x2) / 2) < 1e-9 and y1 == gy, (gx, gy, moves)
      assert x2 == solutions.PASS_PLATE_BACK and y2 == max(gy, solutions.PASS_PLATE_LEFT), \
        (gx, gy, moves)


def _put_down(monkeypatch, run: int, spot: dict, env: dict,
              steps: list | None = None, kind: str = "set_table_pass") -> tuple[float, float]:
  """Where the pass reference's run `run` (0 the cup, 1 the plate, 2 the
  fork) aims its put-down -- or the kitchen shelf's (`kind`), its hands
  reading held: the first `reach` after its walk to the place,
  its x and y worked out by the language's own evaluator (`lang._eval`)
  from the card's tag read at `spot` (`spot.*`) and the locals of `env`
  (set at the pass), the statements between run as the run sets them --
  an `if` by the language's own truth, every verb it tests doing what it
  said, a `step` moving the card as the stance moves under it (each step's
  ahead and left put in `steps`) and a `spot` reading it where it is. A
  statement this walk does not know fails, never one followed otherwise."""
  from types import SimpleNamespace
  from pluggybot.challenge import solutions
  from pluggybot.lifecycle import HUMANOID_HOME, world_facts
  from pluggybot.procedure import axes, lang
  from pluggybot.procedure import library as lib
  facts = world_facts(HUMANOID_HOME)
  library = lib.Library(facts)
  for source in (*solutions.LIBRARY[kind], *solutions.PROCEDURES[kind]):
    library.define(solutions._def_name(source), source)
  body = library.entries[("lay_cup", "lay_plate", "lay_fork")[run]].procedure.body
  card, seen = dict(spot), dict(spot)
  for k in card:
    monkeypatch.setitem(axes.SENSORS, f"spot.{k}", SimpleNamespace(read=lambda life, k=k: seen[k]))
  if kind == "set_table_shelf":
    # the shelf's put-downs go on only while the hand reads held (`SHELF_HELD`)
    for hand in ("left", "right"):
      monkeypatch.setitem(axes.SENSORS, f"{hand}.held", SimpleNamespace(read=lambda life: 1.0))
  env = dict(env)
  taken = [] if steps is None else steps

  def value(expr):
    return lang._eval(expr, env, SimpleNamespace(), facts)

  def verb(name, args):
    if name == "reach":
      return tuple(value(args[a]) for a in ("x", "y"))
    if name == "step":
      ahead, left = value(args["ahead"]), value(args["left"])
      assert "turn" not in args or value(args["turn"]) == 0, "the put-down's step turns"
      taken.append((ahead, left))
      card["x"], card["y"] = card["x"] - ahead, card["y"] - left
    elif name == "spot":
      seen.update(card)
    else:
      raise AssertionError(f"the put-down's walk meets a {name} it does not follow")
    return None

  def walk(stmts):
    for s in stmts:
      if s[0] == "set":
        env[s[1]] = value(s[2])
      elif s[0] == "verb":
        aim = verb(s[1], s[2])
        if aim is not None:
          return aim
      elif s[0] == "if":
        for cond, then in s[1]:
          if cond[0] == "try":
            verb(cond[2][1], cond[2][2])
            held = not cond[1]  # the verb did what it said
          else:
            held = lang._truth(value(cond))
          if held:
            aim = walk(then)
            if aim is not None:
              return aim
            break
        else:
          aim = walk(s[2])
          if aim is not None:
            return aim
      elif s[0] != "pass":
        raise AssertionError(f"the put-down's walk holds a {s[0]} it does not follow")
    return None
  walked = next(i for i, s in enumerate(body) if s[0] == "call" and s[1] == "to_place")
  aim = walk(body[walked + 1:])
  assert aim is not None, "the run puts nothing down after its walk to the place"
  return aim


def test_the_pass_put_downs_are_aimed_off_the_pass_own_constants(monkeypatch):
  """#682's stage C on #751's cure (the loop of 11-26b, round 1): blind to
  its carrying arm, walk2 walks the carried trips true, and every put-down
  tuned on the seeing walker's errors landed a constant away from its spot
  (SimNotes, "The place laid from the pass"). Each piece's aim is the
  pass's own constant, moved by where dev flights on the cure set it down:
  the cup's in the card's own frame, turned by the card's angle in the
  stance (`PASS_CUP_AT`: toward the robot, to its right), the plate's and
  the fork's in the stance frame (`PASS_PLATE_AT`, `PASS_FORK_AT`: short of
  the tag ahead, left of it). Off the reference's own source, run by the
  language."""
  import math
  from pluggybot.challenge import solutions
  ox, oy, oz = 0.012, -0.007, 0.07
  # cards that put no put-down past its stance's band (each run's own steps
  # are pinned below): the cup's trip faces right of the card, so it sees
  # the card further left than the plate's and the fork's trips do
  for sx, sy, a in ((0.52, 0.06, 0.0), (0.49, 0.05, 0.05), (0.55, 0.08, -0.04)):
    spot = {"x": sx, "y": sy, "z": 0.75, "nx": math.cos(a), "ny": math.sin(a)}
    steps = []
    cx, cy = _put_down(monkeypatch, 0, spot, {"ox": ox, "oy": oy, "oz": oz}, steps)
    ca, cb = solutions.PASS_CUP_AT
    assert not steps, (sy, a, steps)
    assert math.isclose(cx, sx - ca * math.cos(a) + cb * math.sin(a) - ox, abs_tol=1e-9), (cx, a)
    assert math.isclose(cy, sy - ca * math.sin(a) - cb * math.cos(a) - oy, abs_tol=1e-9), (cy, a)
  for sx, sy, a in ((0.52, 0.0, 0.0), (0.49, -0.015, 0.05), (0.55, 0.012, -0.04)):
    spot = {"x": sx, "y": sy, "z": 0.75, "nx": math.cos(a), "ny": math.sin(a)}
    for run, at in ((1, solutions.PASS_PLATE_AT), (2, solutions.PASS_FORK_AT)):
      steps = []
      x, y = _put_down(monkeypatch, run, spot, {"oz": oz}, steps)
      assert not steps and math.isclose(x, sx - at[0], abs_tol=1e-9) and \
        math.isclose(y, sy + at[1], abs_tol=1e-9), (run, x, y, at, steps)


def test_the_pass_plate_steps_its_card_ahead_where_its_put_down_comes_near_the_middle(monkeypatch):
  """#682's stage C on #751's cure (the loop of 11-26b, round 10): the walk
  to the place ends with a step forward that leaves the card 2-5 cm across,
  and where the plate's hand was to go under 0.045 m left of the stance's
  middle its reach was out of reach, or took a 0.95 rad bow with the waist
  turned 0.5-0.8 rad and dropped the plate or fell -- 7 of 21 such
  put-downs, where over it no such bow came in 297 (SimNotes, "The place
  laid from the pass"). Looked at after the walk, a card that puts the
  plate's hand nearer the middle than `PASS_PLATE_LEAST` is stepped ahead
  of the stance and looked at again; one that does not is put down as
  read. Off the reference's own source, run by the language."""
  from pluggybot.challenge import solutions
  a, b = solutions.PASS_PLATE_AT

  def stepped(sy):
    steps = []
    x, y = _put_down(monkeypatch, 1, {"x": 0.5, "y": sy, "z": 0.75}, {"oz": 0.07}, steps)
    assert len(steps) == 1 and abs(steps[0][0]) < 1e-12 and abs(steps[0][1] - sy) < 1e-12, (sy, steps)
    assert abs(x - (0.5 - a)) < 1e-9 and abs(y - b) < 1e-9, (sy, x, y)
  for sy in (-0.045 - b, -0.035):
    stepped(sy)
  least = solutions.PASS_PLATE_LEAST
  stepped(least - b - 0.002)
  for sy in (least - b + 0.002, 0.0, 0.06):
    steps = []
    x, y = _put_down(monkeypatch, 1, {"x": 0.5, "y": sy, "z": 0.75}, {"oz": 0.07}, steps)
    assert not steps and abs(y - (sy + b)) < 1e-9, (sy, steps, y)


def test_the_pass_cup_steps_right_where_its_put_down_goes_too_far_right(monkeypatch):
  """#682's stage C on #751's cure (the loop of 11-26b, round 4): where the
  cup's hand was to go further right of the stance's middle than about
  0.20 m, its reach took no waist turn and its pull down crouched, the feet
  swivelled 3-7 deg under the posture's change, and the cup came down 3-5
  cm off its place: 25 of 44 such put-downs within 2.5 cm, where nearer,
  0.11-0.19 m, all 75 were (SimNotes, "The place laid from the pass").
  Looked at after the walk, a card that puts the cup's hand further right
  than `PASS_CUP_RIGHT` is stepped toward, sideways, so that the hand goes
  `PASS_CUP_MID` right, and looked at again; one that does not is put down
  as read. Off the reference's own source, run by the language."""
  import math
  from pluggybot.challenge import solutions
  ca, cb = solutions.PASS_CUP_AT
  ox, oy = 0.012, -0.007
  env = {"ox": ox, "oy": oy, "oz": 0.07}

  def aim(sx, sy, a):
    return (sx - ca * math.cos(a) + cb * math.sin(a) - ox, sy - ca * math.sin(a) - cb * math.cos(a) - oy)
  for sy, a in ((-0.02, 0.0), (0.02, 0.06), (0.03, -0.03)):
    spot = {"x": 0.5, "y": sy, "z": 0.75, "nx": math.cos(a), "ny": math.sin(a)}
    assert aim(0.5, sy, a)[1] < -solutions.PASS_CUP_RIGHT
    steps = []
    x, y = _put_down(monkeypatch, 0, spot, env, steps)
    assert len(steps) == 1 and abs(steps[0][0]) < 1e-12, (sy, steps)
    assert math.isclose(steps[0][1], aim(0.5, sy, a)[1] + solutions.PASS_CUP_MID, abs_tol=1e-12), (sy, steps)
    assert math.isclose(y, -solutions.PASS_CUP_MID, abs_tol=1e-9), (sy, y)
    assert math.isclose(x, aim(0.5, sy, a)[0], abs_tol=1e-9), (sy, x)
  for sy, a in ((0.034, 0.0), (0.06, 0.0), (0.05, 0.05), (0.12, -0.04)):
    spot = {"x": 0.5, "y": sy, "z": 0.75, "nx": math.cos(a), "ny": math.sin(a)}
    assert aim(0.5, sy, a)[1] >= -solutions.PASS_CUP_RIGHT
    steps = []
    x, y = _put_down(monkeypatch, 0, spot, env, steps)
    assert not steps and math.isclose(y, aim(0.5, sy, a)[1], abs_tol=1e-9), (sy, steps, y)


def test_the_pass_fork_steps_left_where_its_put_down_goes_too_far_left(monkeypatch):
  """#682's stage C on #751's cure (the loop of 11-26b, round 4): past about
  0.18 m left of the stance's middle the fork's pull down crouched and
  swivelled the feet as the cup's did -- 25 of 37 put-downs within 2.5 cm
  there, 54 of 57 at 0.11-0.17 m (SimNotes, "The place laid from the
  pass"). A card that puts the fork's hand further left than
  `PASS_FORK_LEFT` is stepped toward, sideways, so that the hand goes
  `PASS_FORK_MID` left, and looked at again; one that does not is put down
  as read. Off the reference's own source, run by the language."""
  from pluggybot.challenge import solutions
  a, b = solutions.PASS_FORK_AT
  for sy in (0.02, 0.05, solutions.PASS_FORK_LEFT - b + 0.002):
    steps = []
    x, y = _put_down(monkeypatch, 2, {"x": 0.5, "y": sy, "z": 0.75}, {"oz": 0.07}, steps)
    assert len(steps) == 1 and abs(steps[0][0]) < 1e-12, (sy, steps)
    assert abs(steps[0][1] - (sy + b - solutions.PASS_FORK_MID)) < 1e-12, (sy, steps)
    assert abs(x - (0.5 - a)) < 1e-9 and abs(y - solutions.PASS_FORK_MID) < 1e-9, (sy, x, y)
  for sy in (solutions.PASS_FORK_LEFT - b - 0.002, 0.0, -0.04):
    steps = []
    x, y = _put_down(monkeypatch, 2, {"x": 0.5, "y": sy, "z": 0.75}, {"oz": 0.07}, steps)
    assert not steps and abs(y - (sy + b)) < 1e-9, (sy, steps, y)


def _cup_set_down(monkeypatch, card: dict, moved: dict | None, env: dict, seen: bool = True) -> list:
  """The pass reference's cup run from its put-down's reach to its let-go,
  the language's own evaluator working each argument out (`lang._eval`): the
  card read at `card` until the reach, then -- the stance turned under the
  held cup by the reach and the pull down -- at `moved` (`seen` False: not
  in the picture), the hand's sensed place where the reach sent it. Each
  verb's name and worked-out arguments, in order, the reach first."""
  from types import SimpleNamespace
  from pluggybot.challenge import solutions
  from pluggybot.lifecycle import HUMANOID_HOME, world_facts
  from pluggybot.procedure import axes, lang
  from pluggybot.procedure import library as lib
  facts = world_facts(HUMANOID_HOME)
  library = lib.Library(facts)
  for source in (*solutions.LIBRARY["set_table_pass"], *solutions.PROCEDURES["set_table_pass"]):
    library.define(solutions._def_name(source), source)
  body = library.entries["lay_cup"].procedure.body
  now = dict(card)
  hand = {}
  for k in ("x", "y", "z", "nx", "ny"):
    monkeypatch.setitem(axes.SENSORS, f"spot.{k}", SimpleNamespace(read=lambda life, k=k: now[k]))
  for k in ("x", "y", "z"):
    monkeypatch.setitem(axes.SENSORS, f"right.{k}", SimpleNamespace(read=lambda life, k=k: hand[k]))
  env = dict(env)
  calls = []

  def value(expr):
    return lang._eval(expr, env, SimpleNamespace(), facts)

  def verb(name, args):
    got = {k: v[1] if isinstance(v, tuple) and v[0] == "str" else value(v) for k, v in args.items()}
    calls.append((name, got))
    if name == "reach":
      hand.update(x=got["x"], y=got["y"], z=got["z"] - 0.09)
      if moved is not None:
        now.update(moved)
    elif name == "move_hand":
      hand.update(x=got["x"], y=got["y"], z=got["z"])
    return name != "spot" or seen

  def walk(stmts):
    for st in stmts:
      if st[0] == "set":
        env[st[1]] = value(st[2])
      elif st[0] == "verb":
        verb(st[1], st[2])
        if st[1] == "open_hand":
          return True
      elif st[0] == "if":
        for cond, then in st[1]:
          if cond[0] == "try":
            ok = verb(cond[2][1], cond[2][2])
            held = (not ok) if cond[1] else ok
          else:
            held = lang._truth(value(cond))
          if held:
            if walk(then):
              return True
            break
        else:
          if walk(st[2]):
            return True
      elif st[0] != "pass":
        raise AssertionError(f"the cup's set-down holds a {st[0]} this walk does not follow")
    return False
  last_reach = max(i for i, st in enumerate(body) if st[0] == "verb" and st[1] == "reach")
  for k in ("tx", "ty", "tz", "a", "ox", "oy", "oz"):
    env.setdefault(k, 0.0)
  # the put-down's aim as the run worked it out, then everything from its reach on
  a = math.atan2(card["ny"], card["nx"])
  ca, cb = solutions.PASS_CUP_AT
  env.update(a=a, tx=card["x"] - ca * math.cos(a) + cb * math.sin(a) - env["ox"],
             ty=card["y"] - ca * math.sin(a) - cb * math.cos(a) - env["oy"], tz=0.89)
  assert walk(body[last_reach:]), "the cup is never let go"
  return calls


def test_the_pass_cup_set_down_is_looked_at_again_and_moved_where_the_stance_turned(monkeypatch):
  """#682's stage C on #751's cure (the loop of 11-26b, round 5): the reach
  and the pull down turned the stance under the held cup 1-2.5 deg, each
  degree 0.6 cm at the cup's place, and a stance stepped sideways first
  turned it 2.5 deg: the cup came down 2-3.5 cm off (SimNotes, "The place
  laid from the pass"). Set down and still held, the cup's run looks at
  the card again; where the cup's place it works out lies `PASS_CUP_AGAIN`
  or further from where the hand was sent, it lifts the cup 1.5 cm, moves
  the hand by that much, sets it down again and lets go; nearer, or the
  card not in the picture, it lets go as it is. Off the reference's own
  source, run by the language."""
  from pluggybot.challenge import solutions
  card = {"x": 0.47, "y": 0.06, "z": 0.775, "nx": math.cos(0.05), "ny": math.sin(0.05)}
  env = {"ox": 0.004, "oy": -0.006, "oz": 0.075}
  ca, cb = solutions.PASS_CUP_AT

  def place(c):
    a = math.atan2(c["ny"], c["nx"])
    return c["x"] - ca * math.cos(a) + cb * math.sin(a), c["y"] - ca * math.sin(a) - cb * math.cos(a)
  # the stance turned 2.5 deg right under the cup: the card seen turned and moved the other way
  d = math.radians(2.5)
  turned = {"x": card["x"] * math.cos(d) - card["y"] * math.sin(d),
            "y": card["x"] * math.sin(d) + card["y"] * math.cos(d), "z": card["z"],
            "nx": math.cos(0.05 + d), "ny": math.sin(0.05 + d)}
  calls = _cup_set_down(monkeypatch, card, turned, env)
  names = [n for n, _ in calls]
  assert names[:3] == ["reach", "pull", "spot"], names
  assert names[3:7] == ["pull", "move_hand", "pull", "open_hand"], names
  (x0, y0), (x1, y1) = place(card), place(turned)
  up, move, down = calls[3][1], calls[4][1], calls[5][1]
  assert up["dz"] == 1 and math.isclose(up["distance"], 0.015) and down["dz"] == -1, (up, down)
  reach = calls[0][1]
  assert math.isclose(move["x"] - reach["x"], x1 - x0, abs_tol=1e-9), (move, reach)
  assert math.isclose(move["y"] - reach["y"], y1 - y0, abs_tol=1e-9), (move, reach)
  assert math.hypot(x1 - x0, y1 - y0) > solutions.PASS_CUP_AGAIN
  # within PASS_CUP_AGAIN, or the card not in the picture: let go as it is
  for moved, seen in ((None, True), (turned, False)):
    names = [n for n, _ in _cup_set_down(monkeypatch, card, moved, env, seen=seen)]
    assert names == ["reach", "pull", "spot", "open_hand"], (moved is None, seen, names)
  small = math.radians(0.8)
  near = {"x": card["x"] * math.cos(small) - card["y"] * math.sin(small),
          "y": card["x"] * math.sin(small) + card["y"] * math.cos(small), "z": card["z"],
          "nx": math.cos(0.05 + small), "ny": math.sin(0.05 + small)}
  assert math.hypot(*(p - q for p, q in zip(place(near), place(card)))) < solutions.PASS_CUP_AGAIN
  names = [n for n, _ in _cup_set_down(monkeypatch, card, near, env)]
  assert names == ["reach", "pull", "spot", "open_hand"], names


def test_the_pass_plate_trip_squares_itself_to_the_pass_before_it_walks_up_to_the_plate():
  """#682's stage C on #751's cure (the loop of 11-26b, round 5): the walk to
  the pass's `face` stops 5.6 deg short of square (median), and a plate
  carried off from a stance turned right of square 8 deg or more was swung
  by the walk's hand-over into the pass's top and came loose in the jaws for
  an instant (SimNotes, "The place laid from the pass"). The plate's run
  turns left `PASS_PLATE_TURN` after its walk to the pass and before it
  walks up to the plate. Off the reference's own source."""
  from pluggybot.challenge import solutions
  calls = _calls(solutions.PROCEDURES["set_table_pass"][1])
  walked = _index(calls, lambda c: c[0] == "to_pass", "walk to the pass")
  up = _index(calls, lambda c: c[0] == "near", "walk up to the plate", walked)
  between = calls[walked + 1:up]
  assert between == [("step", [0, 0, solutions.PASS_PLATE_TURN])], between
  assert 0.05 < solutions.PASS_PLATE_TURN < 0.15, solutions.PASS_PLATE_TURN
  for run in (0, 2):
    calls = _calls(solutions.PROCEDURES["set_table_pass"][run])
    walked = _index(calls, lambda c: c[0] == "to_pass", "walk to the pass")
    assert calls[walked + 1][0] == "near", (run, calls[walked + 1])


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

def test_the_pass_fork_is_brought_back_from_the_pass_in_one_short_pull():
  """#682's stage C: the fork gripped at the pass (`near` lands it 0.43-0.47
  m out) is pulled back before the walk. Pulled back 0.24 m or more in one
  pull, the pull took a 31 deg bow and a 20 cm crouch for its way, stood up
  out of it within 0.3 s as its arm missed, and the robot fell in 6 of 21
  (a shorter pull in none of 79; round 4). Pulled on from `PASS_FORK_BACK`
  to 0.21 m in a second pull (round 4's), the gripped fork's pull took a
  bow of its own about once in 300 fork legs and the body bore on it until
  its feet moved, and the walk's hand-over in that bow fell -- where on
  #751's cure, the hand-over judged by where its hand comes down, its way
  clears the pass's top as high from 0.33 m as from 0.21 (the loop of
  11-26b, round 10; SimNotes, "The place laid from the pass"). Off the
  reference's own source: after the lift off the pass, one pull back to
  `PASS_FORK_BACK`, under 0.24 m from the furthest grip, then the walk."""
  from pluggybot.challenge import solutions
  moves = _hand_moves(solutions.PROCEDURES["set_table_pass"][2], "left")
  shut = _index(moves, lambda m: m[0] == "close_hand", "shut on the fork at the pass")
  lift = _index(moves, lambda m: m[0] == "pull", "lift off the pass", start=shut + 1)
  put = _index(moves, lambda m: m[0] == "reach", "put-down at the place", start=lift + 1)
  back = moves[lift + 1:put]
  assert [m[0] for m in back] == ["go"], f"the fork is brought back by {back}"
  assert float(back[0][1][1]) == solutions.PASS_FORK_BACK, back
  assert 0.47 - solutions.PASS_FORK_BACK < 0.24, solutions.PASS_FORK_BACK


# ---- the place laid from the kitchen's shelf (#682's stage C, the full task) ----------------


def test_the_shelf_reference_is_a_library_the_served_language_takes():
  """The place laid from the kitchen's shelf: the procedures its runs call
  and its three runs, a piece each, compile against the served pair's
  language in one library -- each source inside the language's cap -- and
  only the runs are runnable alone."""
  from pluggybot.challenge import solutions
  from pluggybot.lifecycle import HUMANOID_HOME, world_facts
  from pluggybot.procedure import lang
  from pluggybot.procedure import library as lib
  library = lib.Library(world_facts(HUMANOID_HOME))
  helpers = solutions.LIBRARY["set_table_shelf"]
  runs = solutions.PROCEDURES["set_table_shelf"]
  for source in (*helpers, *runs):
    assert len(source) <= lang.MAX_SOURCE_CHARS
    library.define(solutions._def_name(source), source)
  assert [solutions._def_name(s) for s in runs] == ["lay_cup", "lay_plate", "lay_fork"]
  assert all(e.valid for e in library.entries.values())
  assert set(library.runnable()) == {"lay_cup", "lay_plate", "lay_fork"}


def test_the_shelf_lays_each_piece_as_the_pass_lays_it():
  """Each run of the shelf's, from its walk to the place on, is the pass's
  run byte for byte but its check that the piece is still held, the cup's
  re-set's move taken back where it reads false (`SHELF_CUP_UNDO`), and the
  line past which the fork's put-down steps first (`SHELF_FORK_PUT`): the
  shelf changes how a piece is fetched, that one failure's answer, and that
  one line."""
  from pluggybot.challenge import solutions
  undo_old, undo_new = solutions.SHELF_CUP_UNDO
  far = f'  if read("spot.y") + {solutions._aim(solutions.PASS_FORK_AT[1])} > %g:\n'
  for run, (shelf, pass_, hand) in enumerate(zip(solutions.PROCEDURES["set_table_shelf"],
                                                 solutions.PROCEDURES["set_table_pass"],
                                                 ("right", "left", "left"))):
    assert shelf.count("  to_place(") == pass_.count("  to_place(") == 1
    lay = shelf[shelf.index("  to_place("):]
    assert lay.count(solutions.SHELF_HELD % hand) == 1, hand
    if hand == "right":
      assert lay.count(undo_new) == 1 and pass_.count(undo_old) == 1
      lay = lay.replace(undo_new, undo_old)
    if run == 2:
      shelf_far, pass_far = far % solutions.SHELF_FORK_PUT, far % solutions.PASS_FORK_LEFT
      assert lay.count(shelf_far) == 1 and pass_.count(pass_far) == 1, (shelf_far, pass_far)
      lay = lay.replace(shelf_far, pass_far)
    assert lay.replace(solutions.SHELF_HELD % hand, "") == pass_[pass_.index("  to_place("):]
  assert solutions.LIBRARY["set_table_shelf"][1:] == solutions.LIBRARY["set_table_pass"][1:]


def test_the_shelf_walk_stands_where_a_look_reads_every_piece():
  """On the shelf's 0.96 m board a piece's tag is in `spot`'s picture only
  where it stands lower than `SPOT_TOP_AT_FEET_M - SPOT_TOP_DROP x` over the
  floor, x m ahead of the soles' middle -- the cup's lid within about 0.6 m.
  From the pass's 0.90 m every look failed, and walked to 2.0 m behind the
  pass's tags every first look did (A2's flights; SimNotes, "The place laid
  from the kitchen's shelf"). So the walk to the shelf stands where, arrived
  as far short as a walk arrives, every piece of every set-out reads its
  whole tag at the first look -- and where its planner keeps it off the
  shelf no more than it keeps a walk off any wall."""
  from pluggybot.activity import dishware as dw
  from pluggybot.challenge import solutions
  from pluggybot.evaluation import counter as ct
  from pluggybot.home import areas
  from pluggybot.humanoid.body import WALK_KEEP_M
  from pluggybot.navigator import WAYPOINT_REACHED_M
  from pluggybot.procedure import steps
  tags = {t: (x, y) for t, x, y, _ in areas.area_tags()["restaurant"]}
  wall_x = (tags[68][0] + tags[69][0]) / 2
  front, inward = ct._shelf()
  assert inward[0] > 0.99, "the shelf faces the pass, across the kitchen"
  goal_x = wall_x + solutions.SHELF_ACROSS
  assert front[0] - goal_x >= WALK_KEEP_M, "the walk's goal inside the planner's keep-off"
  tall = {"plate": dw.PLATE_FLOOR_T, "cup": dw.CUP_H, "fork": 2 * dw.HANDLE[2]}
  for k in range(32):
    for piece, (x, _, _) in ct.shelf_set_out(k).pieces.items():
      z = ct.SHELF_H + tall[piece]
      reads = (steps.SPOT_TOP_AT_FEET_M - z) / steps.SPOT_TOP_DROP
      far = x + dw.TAG_SIZE[piece] / 2
      for short in (0.0, WAYPOINT_REACHED_M):
        soles = goal_x - short + ct._stance_ahead()
        assert far - soles <= reads, (k, piece, round(far - soles, 3), round(reads, 3))


def _sweep_parts(roll: float, elbow: float, edge: float, steps_: int = 40):
  """The left arm's pre-pose on the robot's model of itself, standing: its
  shoulder rolled out to `roll`, then its elbow bent to `elbow`, each a
  straight sweep -- whether any corner of the forearm, the wrist or the
  gripper comes beyond `edge` m ahead of the pelvis at or under the shelf's
  0.96 m board, and the gripper's lowest and highest corners at the end."""
  import mujoco
  import numpy as np
  from pluggybot.evaluation import counter as ct
  m = mujoco.MjModel.from_xml_path("models/humanoid.xml")
  d = mujoco.MjData(m)
  parts = [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("left_")
           and any(k in m.geom(g).name for k in ("jaw", "gripper", "pad", "wrist", "forearm"))]
  hand = [g for g in parts if any(k in m.geom(g).name for k in ("jaw", "gripper", "pad"))]

  def corners(q, gs):
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.732
    for name, v in q.items():
      d.qpos[m.jnt_qposadr[m.joint(name).id]] = v
    mujoco.mj_forward(m, d)
    out = []
    for g in gs:
      r, h = d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
      ext = h if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else np.array([h[0], h[0], h[1]])
      for sx in (-1, 1):
        for sy in (-1, 1):
          for sz in (-1, 1):
            out.append(d.geom_xpos[g] + r @ (np.array([sx, sy, sz]) * ext))
    return np.array(out)
  q, met = {}, False
  for joint, target in (("left_shoulder_roll", roll), ("left_elbow", elbow)):
    for i in range(1, steps_ + 1):
      q[joint] = target * i / steps_
      p = corners(q, parts)
      met |= bool(((p[:, 0] > edge) & (p[:, 2] < ct.SHELF_H + 0.005)).any())
  p = corners(q, hand)
  return met, float(p[:, 2].min()), float(p[:, 2].max())


def test_the_shelf_left_pre_pose_clears_the_board_s_front_edge():
  """The left hand's pre-pose before the plate's and the fork's reaches: the
  pass's (shoulder roll 0.9, elbow 1.4) holds the gripper 0.40 m ahead of the
  pelvis, 0.93-0.97 m up, on the shelf board's front edge -- two forks of
  four stopped short there, "caught on something" (A2's flights). Rolled out
  to `SHELF_LEFT_ROLL` before it bends to `SHELF_LEFT_ELBOW`, no corner of
  the arm comes beyond an edge 0.33 m ahead at or under the board on the way,
  and the gripper ends over the board and under the one above. Off the
  shelf's runs' own sources, in that order, before the left hand's reach."""
  from pluggybot.challenge import solutions
  from pluggybot.evaluation import counter as ct
  from pluggybot.home import world as hw
  above = hw.SHELF_BOARDS_Z[ct.SHELF_BOARD + 1] - hw.SHELF_T / 2
  met, low, high = _sweep_parts(solutions.SHELF_LEFT_ROLL, solutions.SHELF_LEFT_ELBOW, 0.33)
  assert not met and low > ct.SHELF_H + 0.02 and high < above - 0.1, (met, low, high)
  assert _sweep_parts(0.9, 1.4, 0.33)[0], "the pass's pre-pose meets the edge"
  for run in solutions.PROCEDURES["set_table_shelf"][1:]:
    calls = _calls(run)
    roll = _index(calls, lambda c: c == ("move", ["left_shoulder_roll", solutions.SHELF_LEFT_ROLL]),
                  "left shoulder rolled out")
    bend = _index(calls, lambda c: c == ("move", ["left_elbow", solutions.SHELF_LEFT_ELBOW]),
                  "left elbow bent", start=roll + 1)
    _index(calls, lambda c: c[0] == "reach" and c[1][:1] == ["left"], "left reach", start=bend + 1)
    assert not any(c[0] == "move" and c[1][1] in (0.9, 1.4) for c in calls), calls


def test_the_shelf_cup_retries_bring_the_hand_out_from_under_the_board_above():
  """A cup whose descent is held back on its lid (or whose reach to its hover
  fails) is withdrawn and looked at again; at the shelf the way home from the
  hover rose into the board above (13-39 N) and was held there (A2's
  flights). Off the reference's own source: every withdraw before the cup is
  gripped first brings the hand straight back to `SHELF_CUP_BACK` ahead."""
  from pluggybot.challenge import solutions
  moves = _hand_moves(solutions.PROCEDURES["set_table_shelf"][0], "right")
  shut = _index(moves, lambda m: m[0] == "close_hand", "shut on the cup")
  withdraws = [i for i in range(shut) if moves[i][0] == "withdraw"]
  assert len(withdraws) == 2, moves[:shut]
  for i in withdraws:
    back = moves[i - 1]
    assert back[0] == "move_hand" and float(back[1][1]) == solutions.SHELF_CUP_BACK, back
  assert solutions.SHELF_CUP_BACK < 0.47 - 0.06, "in front of the shelf's face"


def test_the_shelf_walk_comes_in_straight_before_it_faces_the_shelf():
  """Walked from the staff door straight to its stance, the robot arrived
  turned about 40 deg off the shelf, and the `face` after it stopped 5-13 deg
  short -- the cup outside the picture's side (A2's flights). The walk stops
  `SHELF_FIRST` behind the pass's tags, then walks the rest straight in along
  the walls' normal, and faces the shelf after it."""
  from pluggybot.challenge import solutions
  calls = _calls(solutions.LIBRARY["set_table_shelf"][0])
  walks = [i for i, (n, _) in enumerate(calls) if n == "drive_to"]
  assert len(walks) == 2, calls
  last = round(solutions.SHELF_ACROSS - solutions.SHELF_FIRST, 4)
  assert last >= 0.5, "a last leg too short to square the walk"
  assert calls[walks[1]][1][:2] == [f"mx - {last:g} * cos(f)", f"my - {last:g} * sin(f)"], calls
  verbs = [n for n, _ in calls[walks[1] + 1:] if n in ("face", "step", "drive_to", "spot", "find")]
  assert verbs[:1] == ["face"], verbs


def test_the_shelf_cup_is_read_before_its_pre_pose():
  """Stood 0.43 m from the cup, the right hand's pre-pose put its gripper
  between the camera and the cup's lid and the look after it failed, ending
  the run (A2's flights). Off the reference's own source: each look that
  places the cup's grip comes before the pre-pose's moves, and the look after
  them is an `if`'s test, reading the cup again only where it is seen."""
  import ast as _ast
  from pluggybot.challenge import solutions
  tree = _ast.parse(solutions.PROCEDURES["set_table_shelf"][0])
  body = []
  for node in _ast.walk(tree):
    for field in ("body", "orelse"):
      block = getattr(node, field, None)
      if isinstance(block, list):
        body.append(block)
  seen = 0
  for block in body:
    for i, st in enumerate(block):
      if (isinstance(st, _ast.Expr) and isinstance(st.value, _ast.Call)
          and getattr(st.value.func, "id", "") == "move"
          and st.value.args[0].value == "right_shoulder_roll"):
        seen += 1
        before = [_ast.unparse(s) for s in block[:i]]
        assert any("spot(91)" in b for b in before), before
        after = block[i:]
        tests = [s for s in after if isinstance(s, _ast.If)
                 and _ast.unparse(s.test) == "spot(91)"]
        assert tests, [_ast.unparse(s) for s in after]
        assert not any(_ast.unparse(s) == "spot(91)" for s in after), "a bare look after the pre-pose"
  assert seen == 2, seen


def test_the_shelf_walk_stands_each_piece_where_near_puts_it_and_near_stands_further_off():
  """`near`'s step goes past what it asks -- ahead a median 1.2-3.6 cm, at most
  5.6; across, twice an oblique ask (A2's flights) -- and stood that near,
  the board's edge met the left pre-pose (125 N) and the cup's lid left the
  picture's side. So each walk to the shelf stands where the trip's piece
  lies within 5 cm of where `near` puts it across (`SHELF_S`: the shelf's
  middle off the pass's, from the house, plus the piece's place in the row;
  the fork's 5 cm short, as flown), and `near` stands `SHELF_NEAR_BACK`
  further off than the pass's."""
  import pytest
  from pluggybot.challenge import solutions
  from pluggybot.evaluation import counter as ct
  from pluggybot.home import world as hw
  shelf_y = next(s for s in hw.SHELVES if s[0] == ct.SHELF)[2]
  left_of_pass = shelf_y - sum(hw.EXPO_Y) / 2          # facing east, the left is +y
  pass_near = {n: float(a) for n, (_, a, _) in
               ((r, next(c for c in _calls(src) if c[0] == "near")[1])
                for r, src in zip(("cup", "plate", "fork"), solutions.PROCEDURES["set_table_pass"]))}
  for piece, src in zip(("cup", "plate", "fork"), solutions.PROCEDURES["set_table_shelf"]):
    calls = _calls(src)
    walk = next(c for c in calls if c[0] == "to_shelf")
    assert walk[1] == [solutions.SHELF_S[piece]], walk
    tag, a, b = (float(v) for v in next(c for c in calls if c[0] == "near")[1])
    assert a == pytest.approx(pass_near[piece] + solutions.SHELF_NEAR_BACK), (piece, a)
    row = ct.SHELF_SOURCES[piece][0][1]
    assert abs(solutions.SHELF_S[piece] - (left_of_pass + row - b)) <= 0.05 + 1e-9, piece


def test_the_shelf_cup_s_and_fork_s_missed_looks_turn_toward_them_and_look_again():
  """After `near`, a look that misses the cup or the fork turns
  `SHELF_LOOK_TURN` toward it on the spot -- the cup's row lies right of the
  shelf's middle, the fork's left -- and looks again: `near`'s oblique step
  had left the cup 35 deg off the facing, outside the picture's side (A2's
  flights)."""
  import ast as _ast
  from pluggybot.challenge import solutions
  runs = solutions.PROCEDURES["set_table_shelf"]
  for src, tag, sign in ((runs[0], 91, "-"), (runs[2], 92, "")):
    fn = _ast.parse(src).body[0]
    i = next(j for j, st in enumerate(fn.body) if "near(" in _ast.unparse(st))
    guard = fn.body[i + 1]
    assert isinstance(guard, _ast.If) and _ast.unparse(guard.test) == f"not spot({tag})", _ast.unparse(guard)
    said = [_ast.unparse(st) for st in guard.body]
    assert any(f"step(0, 0, {sign}{solutions.SHELF_LOOK_TURN:g})" in t for t in said), said
    assert said[-1] == f"spot({tag})", said


def test_the_shelf_fork_s_hand_stays_left_of_the_middle_from_its_pull_off_the_shelf():
  """Its hand near the body's middle (0.13-0.20 m left), the fork's pull off
  the shelf, its move back and its stow took a 0.55-0.95 rad bow and the
  waist up to 60 deg in 10 of 49 trips -- the head on the work board in 7,
  one fall -- and from 0.23 m left or more none of 7 (A2's flights). Off the
  reference's own source: the pull off the shelf goes to the hand's own
  left, at least `SHELF_FORK_LEFT`, and every move after it keeps that."""
  import ast as _ast
  from pluggybot.challenge import solutions
  fn = _ast.parse(solutions.PROCEDURES["set_table_shelf"][2]).body[0]
  said = [_ast.unparse(st) for st in fn.body]
  go = next(i for i, t in enumerate(said) if t.startswith("go('left'"))
  least = f"{solutions.SHELF_FORK_LEFT:g}"
  assert said[go - 2:go] == ["y = read('left.y')", f"if y < {least}:\n    y = {least}"], said[go - 2:go]
  assert _ast.unparse(fn.body[go].value.args[2]) == "y", said[go]
  # ...and no move after it, up to the walk, names another left-right place
  walk = next(i for i, t in enumerate(said) if t.startswith("to_place("))
  for t in said[go + 1:walk]:
    for call in [n for n in _ast.walk(_ast.parse(t)) if isinstance(n, _ast.Call)
                 and getattr(n.func, "id", "") == "move_hand"]:
      assert _ast.unparse(call.args[2]) == "read('left.y')", t


def test_the_shelf_puts_each_piece_down_only_while_its_hand_holds_it():
  """After the step at the table let the plate go, the put-down's reach from
  the hanging arm (the walk had brought the empty hand home) bowed the robot
  1.10 rad and it fell (A2's flights). Off the reference's own source: in each
  shelf run the walk to the place is followed at once by the end of the run
  where its hand's grip reads empty (`SHELF_HELD`), before any put-down. In
  60 of A2's trips whose walk said it had dropped its piece (of 576) none was
  laid within 2.5 cm, so the check costs no piece."""
  import ast as _ast
  from pluggybot.challenge import solutions
  for src, hand in zip(solutions.PROCEDURES["set_table_shelf"], ("right", "left", "left")):
    fn = _ast.parse(src).body[0]
    i = next(j for j, st in enumerate(fn.body)
             if isinstance(st, _ast.Expr) and _ast.unparse(st).startswith("to_place("))
    guard = fn.body[i + 1]
    assert isinstance(guard, _ast.If), _ast.unparse(guard)
    assert _ast.unparse(guard.test) == f"read('{hand}.held') < 0.5", _ast.unparse(guard.test)
    assert [type(st) for st in guard.body] == [_ast.Return], _ast.unparse(guard)


def test_the_shelf_s_thin_pieces_are_stowed_clear_of_the_board_before_the_walk():
  """The walk's hand-over takes a thin grip's side way, which rose a median
  0.21 m (at most 0.39) and swung 0.06-0.16 m forward before it came down:
  from over the board the plate's far rim came down on its edge (18 of 58)
  and the fork struck the board above (46 of 52), falling out of the jaws 8
  times (A2's flights). Off the reference's own source: after each thin
  piece's pull-back, its hand is moved back to `SHELF_*_BACK` -- its far end
  then in front of the board's face where `near` stands it -- and lowered to
  `SHELF_STOW_Z`, from which the side way's highest rise stays under the
  board above; then the walk. The fork is lowered by a `pull` that holds
  its line and turn (`give=0`): lowered by `move_hand`, two forks of eight
  turned in the jaws and fell out."""
  from pluggybot.activity import dishware as dw
  from pluggybot.challenge import solutions
  from pluggybot.evaluation import counter as ct
  from pluggybot.home import world as hw
  above = hw.SHELF_BOARDS_Z[ct.SHELF_BOARD + 1] - hw.SHELF_T / 2
  assert solutions.SHELF_STOW_Z + 0.39 < above, "the side way's highest rise strikes the board above"
  assert solutions.SHELF_STOW_Z < ct.SHELF_H, "lowered under the board's top: in front of its face"
  far = {"plate": 2 * dw.PLATE_R - dw.PLATE_WALL_T, "fork": dw.HANDLE[0] + 2 * dw.HEAD[0]}
  back = {"plate": solutions.SHELF_PLATE_BACK, "fork": solutions.SHELF_FORK_BACK}
  for piece, src in zip(("plate", "fork"), solutions.PROCEDURES["set_table_shelf"][1:]):
    calls = _calls(src)
    a = next(c for c in calls if c[0] == "near")[1][1]
    face = a - ct.SHELF_SOURCES[piece][0][0]        # the board's face ahead, where near stands
    assert back[piece] + far[piece] < face, (piece, back[piece] + far[piece], face)
    moves = _hand_moves(src, "left")
    i_back = _index(moves, lambda m: m[0] == "move_hand" and m[1][1] == back[piece],
                    f"{piece} moved back to SHELF's")
    stows = [moves[i_back + 1]]
    if piece == "plate":
      stow = stows[0]
      assert stow[0] == "move_hand" and stow[1][3] == solutions.SHELF_STOW_Z, stow
    else:
      # ...the fork's by one pull, its ask the next test's (`SHELF_FORK_EDGE`)
      lowered = f'read("left.z") - {solutions.SHELF_STOW_Z:g}'
      stows = [m for m in moves[i_back + 1:] if m[0] == "pull" and isinstance(m[1][5], str)
               and m[1][5].replace("'", '"') == lowered]
      assert stows and moves[i_back + 1:i_back + 1 + len(stows)] == stows, moves[i_back + 1:]
      for stow in stows:
        assert [float(v) for v in stow[1][1:4]] == [0, 0, -1], stow
      gives = [k.value.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
               and getattr(n.func, "id", "") == "pull" and ast.unparse(n.args[5]).replace("'", '"') == lowered
               for k in n.keywords if k.arg == "give"]
      assert gives == [0] * len(stows), gives
    # ...and the walk next: no hand move between the stow and the walk to the place
    walk = _index(calls, lambda c: c[0] == "to_place", "walk to the place")
    after = [c for c in calls[:walk] if c[0] in ("move_hand", "go", "pull", "reach")]
    assert after[-len(stows):] == stows, after[-len(stows):]


def test_the_shelf_fork_s_stow_asks_little_only_where_its_tip_stands_at_the_board_s_edge():
  """The fork lies on the shelf with its head toward it, the tip 0.125 m past
  the grip and the fork 0.07 m in from the board's edge: brought back to
  `SHELF_FORK_BACK` from a grip under 0.41 m ahead, the tip stood over the
  edge, and the stow's lowering, asking 20 N, pressed it on the board's top
  and turned the fork 28-57 deg in the jaws in 14 of 15 trips. Asking 3 N it
  stops where the tip meets the board, the fork flat; asked of every stow, 3
  N slowed almost every one as it began and flew almost every fork trip
  afresh (A5's and A6's flights; SimNotes, "The place laid from the kitchen's
  shelf"). Off
  the reference's own source: the grip's place is read where the move back
  begins, no verb between; the lowering asks `SHELF_FORK_STOW_N` where it
  reads nearer than `SHELF_FORK_EDGE` -- the tip then within 1 cm of the edge
  or over it -- and 20 N, as before, where it does not."""
  import ast as _ast
  from pluggybot.activity import dishware as dw
  from pluggybot.challenge import solutions
  from pluggybot.evaluation import counter as ct
  tip, lies = dw.HANDLE[0] + 2 * dw.HEAD[0], ct.SHELF_SOURCES["fork"][0][0]
  assert math.isclose(solutions.SHELF_FORK_EDGE, solutions.SHELF_FORK_BACK + tip + lies + 0.01), (
    "the edge's line is no longer the tip within 1 cm of the board's edge")
  fn = _ast.parse(solutions.PROCEDURES["set_table_shelf"][2]).body[0]
  said = [_ast.unparse(st) for st in fn.body]
  back = f"go('left', {solutions.PASS_FORK_BACK:g},"
  go = next((i for i, t in enumerate(said) if t.startswith(back)), None)
  assert go is not None, f"the fork's run has no move back ({back})"
  read = said.index("gx = read('left.x')")
  moved = [c.func.id for st in fn.body[read + 1:go] for c in _ast.walk(st)
           if isinstance(c, _ast.Call) and isinstance(c.func, _ast.Name) and c.func.id != "read"]
  assert read < go and not moved, said[read:go + 1]
  stow = next(st for st in fn.body[go:] if isinstance(st, _ast.If)
              and _ast.unparse(st.test).startswith("read('left.z') >"))
  gate = stow.body[0]
  assert len(stow.body) == 1 and isinstance(gate, _ast.If), _ast.unparse(stow)
  assert _ast.unparse(gate.test) == f"gx < {solutions.SHELF_FORK_EDGE:g}", _ast.unparse(gate)

  def asks(body):
    return [float(_ast.literal_eval(c.args[4])) for st in body for c in _ast.walk(st)
            if isinstance(c, _ast.Call) and getattr(c.func, "id", "") == "pull"]
  assert asks(gate.body) == [solutions.SHELF_FORK_STOW_N] and asks(gate.orelse) == [20.0], _ast.unparse(gate)
  assert solutions.SHELF_FORK_STOW_N < 20


def test_the_shelf_fork_steps_left_where_its_put_down_goes_past_the_shelf_s_line(monkeypatch):
  """On the shelf's trips the fork's put-down aimed 0.17-0.18 m left of the
  stance's middle failed 18 of 83 -- its pull down out of reach, or the
  stance turning under it -- where those stepped from 0.18-0.19 m, a step
  going about half its way, failed 1 of 42 (read off round 9's own rows;
  SimNotes, "The place laid from the kitchen's shelf"). A card that puts the
  fork's hand further left than `SHELF_FORK_PUT` is stepped toward, sideways,
  so that the hand goes `PASS_FORK_MID` left, and looked at again; one that
  does not is put down as read: the pass's rule (`PASS_FORK_LEFT`) at the
  shelf's own line. Off the reference's own source, run by the language."""
  from pluggybot.challenge import solutions
  a, b = solutions.PASS_FORK_AT
  assert solutions.PASS_FORK_MID < solutions.SHELF_FORK_PUT < solutions.PASS_FORK_LEFT
  for sy in (solutions.SHELF_FORK_PUT - b + 0.002, solutions.PASS_FORK_LEFT - b - 0.002, 0.05):
    steps = []
    x, y = _put_down(monkeypatch, 2, {"x": 0.5, "y": sy, "z": 0.75}, {"oz": 0.07}, steps,
                     kind="set_table_shelf")
    assert len(steps) == 1 and abs(steps[0][0]) < 1e-12, (sy, steps)
    assert abs(steps[0][1] - (sy + b - solutions.PASS_FORK_MID)) < 1e-12, (sy, steps)
    assert abs(x - (0.5 - a)) < 1e-9 and abs(y - solutions.PASS_FORK_MID) < 1e-9, (sy, x, y)
  for sy in (solutions.SHELF_FORK_PUT - b - 0.002, 0.0, -0.04):
    steps = []
    x, y = _put_down(monkeypatch, 2, {"x": 0.5, "y": sy, "z": 0.75}, {"oz": 0.07}, steps,
                     kind="set_table_shelf")
    assert not steps and abs(y - (sy + b)) < 1e-9, (sy, steps, y)


def _shelf_cup_reset(monkeypatch, card: dict, moved: dict, swung: tuple | None) -> list:
  """The shelf reference's cup run from its put-down's reach to its let-go,
  the language's own evaluator working each argument out (`lang._eval`),
  as `_cup_set_down` runs the pass's: the card read at `card` until the
  reach, then at `moved`; the hand's sensed place where each move sent it --
  but a `move_hand` after the re-set's lift, given `swung`, reads false and
  leaves the hand that far off where it was sent (x, y, m). Each call's name
  and worked-out arguments, a procedure's (`go`) as a verb's, in order."""
  from types import SimpleNamespace
  from pluggybot.challenge import solutions
  from pluggybot.lifecycle import HUMANOID_HOME, world_facts
  from pluggybot.procedure import axes, lang
  from pluggybot.procedure import library as lib
  facts = world_facts(HUMANOID_HOME)
  library = lib.Library(facts)
  for source in (*solutions.LIBRARY["set_table_shelf"], *solutions.PROCEDURES["set_table_shelf"]):
    library.define(solutions._def_name(source), source)
  body = library.entries["lay_cup"].procedure.body
  now = dict(card)
  hand = {}
  for k in ("x", "y", "z", "nx", "ny"):
    monkeypatch.setitem(axes.SENSORS, f"spot.{k}", SimpleNamespace(read=lambda life, k=k: now[k]))
  for k in ("x", "y", "z"):
    monkeypatch.setitem(axes.SENSORS, f"right.{k}", SimpleNamespace(read=lambda life, k=k: hand[k]))
  env = {"ox": 0.004, "oy": -0.006, "oz": 0.075}
  calls, began = [], []

  def value(expr):
    return lang._eval(expr, env, SimpleNamespace(), facts)

  def verb(name, args):
    got = {k: v[1] if isinstance(v, tuple) and v[0] == "str" else value(v) for k, v in args.items()}
    calls.append((name, got))
    if name == "reach":
      hand.update(x=got["x"], y=got["y"], z=got["z"] - 0.09)
      now.update(moved)
    elif name == "pull":
      way = [got[k] for k in ("dx", "dy", "dz")]
      n = math.sqrt(sum(v * v for v in way))
      hand.update(**{k: hand[k] + got["distance"] * v / n for k, v in zip("xyz", way)})
    elif name == "move_hand":
      began.append(dict(hand))
      lifted = len(calls) > 1 and calls[-2][0] == "pull" and calls[-2][1]["dz"] == 1
      if lifted and swung is not None:
        hand.update(x=got["x"] + swung[0], y=got["y"] + swung[1], z=got["z"])
        return False
      hand.update(x=got["x"], y=got["y"], z=got["z"])
    return True

  def walk(stmts):
    for st in stmts:
      if st[0] == "set":
        env[st[1]] = value(st[2])
      elif st[0] in ("verb", "call"):
        verb(st[1], st[2])
        if st[1] == "open_hand":
          return True
      elif st[0] == "if":
        for cond, then in st[1]:
          if cond[0] == "try":
            ok = verb(cond[2][1], cond[2][2])
            held = (not ok) if cond[1] else ok
          else:
            held = lang._truth(value(cond))
          if held:
            if walk(then):
              return True
            break
        else:
          if walk(st[2]):
            return True
      elif st[0] != "pass":
        raise AssertionError(f"the cup's set-down holds a {st[0]} this walk does not follow")
    return False
  last_reach = max(i for i, st in enumerate(body) if st[0] == "verb" and st[1] == "reach")
  a = math.atan2(card["ny"], card["nx"])
  ca, cb = solutions.PASS_CUP_AT
  env.update(a=a, tx=card["x"] - ca * math.cos(a) + cb * math.sin(a) - env["ox"],
             ty=card["y"] - ca * math.sin(a) - cb * math.cos(a) - env["oy"], tz=0.89)
  assert walk(body[last_reach:]), "the cup is never let go"
  return calls, began


def test_the_shelf_cup_s_re_set_move_is_taken_back_where_it_reads_false(monkeypatch):
  """The cup's re-set (the pass's: looked at again set down and still held,
  lifted 1.5 cm and moved where the card puts its place) moved the cup
  further off in 7 of the 8 re-sets whose move read false that can be read:
  out of reach, the move swung the hand 5-8 cm to the right, and the cup came
  down 3-7 cm off a place it had first been set down within 2.5 cm of (A6's
  reading of A5's rows; SimNotes, "The place laid from the kitchen's
  shelf"). Off the
  reference's own source, run by the language: where the re-set's move
  reads false, the hand goes back to where the move began by `go` -- the
  place read as it began, no verb between -- before it sets the cup down;
  where it reads true, the run is the pass's, nothing added."""
  card = {"x": 0.47, "y": 0.06, "z": 0.775, "nx": math.cos(0.05), "ny": math.sin(0.05)}
  d = math.radians(2.5)
  turned = {"x": card["x"] * math.cos(d) - card["y"] * math.sin(d),
            "y": card["x"] * math.sin(d) + card["y"] * math.cos(d), "z": card["z"],
            "nx": math.cos(0.05 + d), "ny": math.sin(0.05 + d)}
  calls, _ = _shelf_cup_reset(monkeypatch, card, turned, swung=None)
  assert [n for n, _ in calls] == ["reach", "pull", "spot", "pull", "move_hand", "pull", "open_hand"], calls
  calls, began = _shelf_cup_reset(monkeypatch, card, turned, swung=(-0.01, -0.07))
  names = [n for n, _ in calls]
  assert names == ["reach", "pull", "spot", "pull", "move_hand", "go", "pull", "open_hand"], names
  up, move, back, down = (calls[i][1] for i in (3, 4, 5, 6))
  assert up["dz"] == 1 and math.isclose(up["distance"], 0.015) and down["dz"] == -1, (up, down)
  # ...back to where the failing move began, the cup lifted: the hand's place as it began
  assert [back[k] for k in ("h", "x", "y", "z")] == ["right", *(began[-1][k] for k in "xyz")], (back, began)
  assert not math.isclose(move["y"], began[-1]["y"], abs_tol=1e-6), "the re-set moved nothing"
