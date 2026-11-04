"""Procedures that call procedures (#560, Ben's decision of 2026-11-04):
a source holds one def or more, the first runs, and a def calls another by
its name -- with arguments, on the run's one budget, a few calls deep --
and in the robot's library one procedure calls another. Every rule pinned
on the parser, the stub life or the library: nothing here flies.
"""

import textwrap
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pluggybot import tick
from pluggybot.lifecycle import errand_from, procedure_outcome, world_facts
from pluggybot.mind import overseer as ov
from pluggybot.mind import practice as mp
from pluggybot.mind.overseer import Decision
from pluggybot.procedure import lang, library as lib, steps as st
from pluggybot.procedure.steps import Refused
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path
from test_language import _stub_life

WORLD = "home_humanoid"
HOME = world_facts(WORLD)
#: the language as it stood before calls, a world that gives none
NO_CALLS = replace(HOME, calls=False)

TURNS = '''def both():
    turn("left_elbow")
    turn("right_elbow", 0.25)
    k = 3

def turn(axis, a=0.5):
    b = a * 2
    move(axis, b)
'''


def compile_ok(src, facts=HOME, **kw):
  return lang.compile_procedure(src, facts, **kw)


def refusals(src, facts=HOME, **kw):
  with pytest.raises(Refused) as e:
    lang.compile_procedure(src, facts, **kw)
  return e.value.reasons


def run(src, life=None, facts=HOME):
  life = life or _stub_life()
  proc = lang.compile_procedure(src, facts)
  r = tick.run(SimpleNamespace(step=lambda *a: None),
               lang.run_procedure_routine(life, proc, facts))
  return r, life


# ---- the grammar ----------------------------------------------------------------


def test_a_source_of_several_defs_compiles_only_where_the_world_gives_calls():
  """Its first def is the procedure; the hands' worlds give calls, the
  practice's and the served humanoids' alike, and a world that gives none
  keeps the one def with no arguments."""
  proc = compile_ok(TURNS)
  assert proc.name == "both" and set(proc.defs) == {"turn"} and proc.depth == 1
  assert proc.verbs == 2, "each call counts its callee's verbs"
  assert proc.defs["turn"].params == ("axis", "a") and proc.defs["turn"].names == {"axis"}
  assert st.hand_facts().calls and HOME.calls
  assert refusals(TURNS, NO_CALLS) == ["a procedure is exactly one `def name():`"]
  assert any("takes no arguments" in r for r in
             refusals("def x(y=1):\n  wait(1)\n", NO_CALLS))


@pytest.mark.parametrize("src, said", [
  ("def a():\n  wait(1)\nx = 1\n", "a source is one def or more"),
  ("def a():\n  b()\ndef b():\n  wait(1)\ndef b():\n  wait(2)\n", "line 5: two defs are named b"),
  ("def a():\n  wait(1)\ndef wait():\n  face(1)\n",
   "line 3: 'wait' is the language's own name: a def takes a name of its own"),
  ("def a():\n  b = 1\n  b()\ndef b():\n  wait(1)\n", "'b' is not a name this language allows"),
  ("def a():\n  x = b()\ndef b():\n  wait(1)\n", "a verb is a statement of its own, and so is a call"),
  ("def a():\n  b(1, 2)\ndef b(x):\n  wait(x)\n", "line 2, b: takes 1 argument(s)"),
  ("def a():\n  b(y=1)\ndef b(x):\n  wait(x)\n", "line 2, b: takes no y"),
  ("def a():\n  b(1, x=2)\ndef b(x):\n  wait(x)\n", "line 2, b: x is given twice"),
  ("def a():\n  b()\ndef b(x):\n  wait(x)\n", "line 2, b: needs x"),
  ("def a():\n  b(3)\ndef b(hand):\n  open_hand(hand)\n", "line 2, b: hand takes a name in quotes"),
  ("def a():\n  b(\"left\")\ndef b(x):\n  wait(x)\n",
   "line 2, b: x takes a number, not a name in quotes"),
  ("def a():\n  b(\"left\")\ndef b(h):\n  wait(h)\n  open_hand(h)\n",
   "line 4: h holds a name (line 5), and is used as a number"),
  ("def a():\n  b(\"left\")\ndef b(h):\n  h = 2\n  open_hand(h)\n",
   "line 4: h holds a name, which is never set"),
  ("def a():\n  b()\ndef b(h=2):\n  open_hand(h)\n", "h holds a name, and its default is a number"),
  ("def a():\n  b()\ndef b(x=\"two\"):\n  wait(x)\n", "x holds a number, and its default is a name"),
  ("def a():\n  b()\ndef b(x=read(\"time\")):\n  wait(x)\n",
   "the default of x is a number or a name in quotes"),
  ("def a():\n  b()\ndef b(*xs):\n  wait(1)\n", "a def's arguments are plain names"),
  ("def a():\n  b()\ndef b(b):\n  wait(1)\n", "'b' is a def's name"),
  ("def a(x):\n  wait(x)\n", "line 1: a is the def that runs, and nothing passes it x"),
  ("def a():\n  open_hand(h)\n", "hand takes a name in quotes, or an argument that holds one: "
                                 "h is no argument of a's"),
])
def test_each_call_rule_is_refused_with_its_line(src, said):
  reasons = refusals(src)
  assert any(said in r for r in reasons), reasons


def test_a_def_may_not_call_itself_through_another_or_not():
  """The language is total: no def calls round to itself, and the loop is
  named once, with no `calls no verb` beside it (its verbs are uncounted)."""
  assert refusals("def a():\n  a()\n") == [
    "line 2: a calls itself: a def may not call itself, by way of another or not"]
  assert refusals("def a():\n  b()\ndef b():\n  c()\ndef c():\n  a()\n") == [
    "line 6: a calls b, which calls c, which calls a: a def may not call itself, "
    "by way of another or not"]


def test_calls_go_at_most_their_cap_deep():
  defs = [f"def d{i}():\n  d{i + 1}()\n" for i in range(lang.MAX_CALL_DEPTH)]
  deep = "".join(defs) + f"def d{lang.MAX_CALL_DEPTH}():\n  wait(1)\n"
  assert compile_ok(deep).depth == lang.MAX_CALL_DEPTH
  deeper = "def top():\n  d0()\n" + deep
  [why] = refusals(deeper)
  assert why == (f"line 1: calls go {lang.MAX_CALL_DEPTH + 1} deep below top "
                 f"(top > d0 > d1 > d2 > d3 > d4); the most is {lang.MAX_CALL_DEPTH}"), why


def test_a_name_a_call_passes_is_checked_against_the_world_where_it_is_used():
  """`put("middle", ...)` is refused before anything moves, at the line
  that uses it and through the calls it came by -- a sensor joined from
  it too, and a move's stop condition."""
  [why] = refusals('def a():\n  b("middle")\ndef b(hand):\n  c(hand)\n'
                   'def c(h):\n  open_hand(h)\n')
  assert why == ("line 6, open_hand: hand='middle' is not one of left, right -- called at "
                 "line 2, b > line 4, c"), why
  [why] = refusals('def a():\n  b("left")\ndef b(h):\n  x = read(h + ".zz")\n  wait(1)\n')
  assert why == "line 4: unknown sensor 'left.zz' -- called at line 2, b", why
  [why] = refusals('def a():\n  b("left")\ndef b(h):\n  reach(h, 0.3, 0.2, 1.0, '
                   'until=h + ".zz", above=1)\n')
  assert why.startswith("line 4, reach: until='left.zz' is no sensor you have"), why
  # ...and each name a call passes is checked at each call
  assert compile_ok('def a():\n  b("left")\n  b("right")\ndef b(h):\n  open_hand(h)\n'
                    '  x = read(h + ".z")\n')


def test_a_source_over_the_cap_is_refused_with_every_other_reason():
  """#615's smoke: three repairs of 5,458 characters each heard only that
  the source was too long, and never of `spot.x` or `spot("93")`. Read
  whole, a source over the cap is refused with every reason at once; one
  cut short of parsing (the served answer is cut one over) with the cap
  alone."""
  pad = "  wait(1)\n" * (lang.MAX_SOURCE_CHARS // 10)
  long = "def x():\n  a = spot.x\n" + pad
  reasons = refusals(long)
  assert reasons[0] == f"the source is {len(long)} characters; the cap is {lang.MAX_SOURCE_CHARS}"
  assert any("spot.x is a sensor" in r for r in reasons), reasons
  assert refusals(long[:lang.MAX_SOURCE_CHARS + 1] + "(") == [reasons[0].replace(
    str(len(long)), str(lang.MAX_SOURCE_CHARS + 2))]
  assert lang.MAX_SOURCE_CHARS == 3000


# ---- the run --------------------------------------------------------------------


def test_a_call_runs_its_def_and_its_variables_ride_the_readout():
  r, life = run(TURNS)
  assert r["ok"] and r["total"] == 2 and [c for c in life.calls if c[0] == "ramp"] == [
    ("ramp", (0, 1.0, 1.0)), ("ramp", (0, 0.5, 1.0))]
  assert [(s["verb"], s["line"], s.get("in")) for s in r["steps"]] == [
    ("move", 8, "turn"), ("move", 8, "turn")]
  assert r["locals"] == {"k": 3.0, "turn.b": 0.5}, "a callee's own, as it last ended"


def test_a_return_in_a_call_returns_from_its_def_alone():
  r, life = run("def a():\n  b()\n  wait(2)\ndef b():\n  wait(1)\n  return\n  wait(9)\n")
  assert r["ok"] and [c[1][0] for c in life.calls] == [1.0, 2.0]


def test_the_run_has_one_budget_and_every_call_spends_from_it():
  r, life = run("def a():\n  budget(steps=3)\n  for i in range(5):\n    b()\n"
                "def b():\n  budget(steps=200)\n  wait(1)\n")
  assert r["stopped"] == "steps" and r["total"] == 3 and len(life.calls) == 3


def test_a_verb_that_fails_in_a_call_ends_the_run_and_says_where():
  life = _stub_life()
  life.body.go_to_routine = lambda *a, **kw: tick.result(False)
  r, _ = run("def a():\n  b()\n  wait(2)\ndef b():\n  wait(1)\n  drive_to(1, 1)\n", life)
  assert not r["ok"] and r["failedAt"] == 1 and r["completed"] == 1
  failed = r["steps"][r["failedAt"]]
  assert (failed["verb"], failed["line"], failed["in"]) == ("drive_to", 6, "b")
  [line] = procedure_outcome("a", r)
  assert "it stopped at line 6, drive_to (in b): did not arrive" in line, line


def test_a_call_an_if_tests_reads_a_failure_in_it_and_the_run_goes_on():
  """As a verb's test: a verb failing in the call, or a fault in its
  arithmetic, makes it false; the run's own limits still end the run."""
  r, life = run("def a():\n  if not b(99):\n    wait(2)\n  wait(3)\n"
                "def b(s):\n  wait(1)\n  wait(s)\n  wait(5)\n")
  assert r["ok"] and [c[1][0] for c in life.calls] == [1.0, 2.0, 3.0]
  read = [s for s in r["steps"] if not s["ok"]]
  assert len(read) == 1 and read[0]["tested"] and read[0]["in"] == "b" and "failedAt" not in r
  r, life = run("def a():\n  if b(0):\n    wait(9)\n  wait(3)\ndef b(s):\n  wait(1)\n"
                "  x = 1 / s\n")
  assert r["ok"] and [c[1][0] for c in life.calls] == [1.0, 3.0]
  assert {"verb": "expr", "ok": False, "reason": "division by zero"}.items() <= next(
    s for s in r["steps"] if not s["ok"]).items()
  r, _ = run("def a():\n  budget(steps=2)\n  if b():\n    pass\n  wait(3)\n"
             "def b():\n  for i in range(5):\n    wait(1)\n")
  assert r["stopped"] == "steps" and not r["ok"]


def test_an_argument_a_run_cannot_compute_fails_its_call_at_its_line():
  r, _ = run("def a():\n  b(z)\ndef b(x):\n  wait(x)\n")
  assert not r["ok"] and r["steps"][-1] == {"i": 0, "verb": "b", "line": 2, "ok": False,
                                            "reason": "'z' was read before it was set"}
  [line] = procedure_outcome("a", r)
  assert "it stopped at line 2, b: 'z' was read before it was set" in line, line


def test_statements_are_capped_where_loops_in_calls_multiply():
  r, life = run("def a():\n  for i in range(100):\n    b()\n  wait(1)\n"
                "def b():\n  for j in range(100):\n    c()\n"
                "def c():\n  for k in range(100):\n    x = k + 1\n")
  assert r["stopped"] == "statements" and not r["ok"] and life.calls == []
  assert "it ran 100,000 statements, the most one run may" in mp.run_report(1, r, {
    "ok": False, "reason": "x"}, 1.0, False)
  [line] = procedure_outcome("a", r)
  assert "it ran the most statements one run may" in line, line


# ---- the library ------------------------------------------------------------------


def test_one_procedure_calls_another_of_the_library():
  L = lib.Library(HOME)
  L.define("turn", "def turn(axis, a=0.5):\n  move(axis, a)\n")
  L.define("both", 'def both():\n  turn("left_elbow")\n  turn("right_elbow", 0.25)\n')
  assert L.runnable() == ("both",) and L.get("both").linked["turn"] is L.get("turn")
  ctx = {r["name"]: r for r in L.as_context()}
  assert ctx["turn"]["runnable"] is False and ctx["turn"]["reasons"] == [
    "it takes axis: it runs only when another procedure calls it"]
  assert errand_from(Decision(action="procedure:turn"), library=L) is None
  r, _ = run_entry(L, "both")
  assert r["ok"] and [(s["in"], s["entry"]) for s in r["steps"]] == [("turn", "turn")] * 2
  with pytest.raises(Refused, match="nothing passes it"):
    lang.compile_procedure(L.sources["turn"], HOME)       # alone, it is no procedure


def run_entry(L, name, life=None):
  life = life or _stub_life()
  return tick.run(SimpleNamespace(step=lambda *a: None),
                  lang.run_procedure_routine(life, L.get(name), HOME)), life


def test_a_caller_whose_callee_went_is_kept_marked_and_mended_when_it_returns(tmp_path):
  root = tmp_path / "procedures"
  L = lib.Library(HOME, root=root)
  L.define("turn", "def turn(axis):\n  move(axis, 0.5)\n")
  L.define("both", 'def both():\n  turn("left_elbow")\n')
  L.undefine("turn")
  assert L.names() == ("both",) and L.runnable() == ()
  assert "unknown verb 'turn'" in L.as_context()[0]["reasons"][0]
  assert "calls no verb" not in " ".join(L.as_context()[0]["reasons"])
  L.define("turn", "def turn(axis):\n  move(axis, 0.25)\n")
  assert L.runnable() == ("both",)
  # ...and a restart compiles each after the ones it calls, whatever the files' order
  again = lib.Library(HOME, root=root)
  assert again.runnable() == ("both",) and again.get("both").linked["turn"].source.endswith(
    "0.25)\n")


def test_a_define_that_would_close_a_loop_through_the_library_is_refused():
  """A caller left waiting for its callee is no procedure to call: the
  callee written to call it back is refused, so no two entries ever call
  round to each other."""
  L = lib.Library(HOME)
  with pytest.raises(lib.LibraryRefused, match="unknown verb 'b'"):
    L.define("a", "def a():\n  b()\n")
  L.define("b", "def b():\n  wait(1)\n")
  L.define("c", "def c():\n  b()\n")
  L.undefine("b")
  with pytest.raises(lib.LibraryRefused, match="c calls b, which calls c: a def may not call "
                                                 "itself"):
    L.define("b", "def b():\n  c()\n")
  assert L.names() == ("c",)


def test_a_def_of_a_procedures_own_source_comes_before_the_librarys():
  L = lib.Library(HOME)
  L.define("turn", "def turn():\n  wait(9)\n")
  L.define("go", "def go():\n  turn()\ndef turn():\n  wait(1)\n")
  r, life = run_entry(L, "go")
  assert r["ok"] and [c[1][0] for c in life.calls] == [1.0]


def test_a_define_or_undefine_says_which_callers_it_stopped_or_mended():
  """History is where the robot reads what its library did (#264): an
  undefine that stops a caller says so, and so does the define that mends
  it, or a replacement whose call no longer fits."""
  life = stub_life(WORLD)
  L = lib.Library(HOME)
  life.overseer = SimpleNamespace(library=L, event_map=None)

  def answer(**kw):
    life._define(Decision(action="idle", reason="r", **kw))
  answer(define={"name": "turn", "source": "def turn(axis, a=0.5):\n  move(axis, a)\n"})
  answer(define={"name": "both", "source": 'def both():\n  turn("left_elbow")\n'})
  answer(undefine="turn")
  answer(define={"name": "turn", "source": "def turn(axis, a=0.5):\n  move(axis, a)\n"})
  answer(undefine="turn", define={"name": "turn",
                                  "source": "def turn(axis, a, b):\n  move(axis, a + b)\n"})
  history = life.thoughts.read("History.md")
  assert ("forgot the procedure turn -- both, which calls turn, cannot run until a turn is "
          "written again") in history
  assert "wrote the procedure turn -- both can run again" in history
  assert "wrote the procedure turn -- both cannot run now: line 2, turn: needs a, b" in history


# ---- what the robot is told -----------------------------------------------------


def test_the_language_says_its_caps_off_the_constants():
  text = mp.language_text(HOME)
  assert f"at most {lang.MAX_SOURCE_CHARS:,} characters" in text
  assert f"calls go at most {lang.MAX_CALL_DEPTH} deep" in text
  assert "So may a call" in text and mp.FIRST_DEF in text and mp.ONE_DEF not in text
  bare = mp.language_text(NO_CALLS)
  assert f"at most {lang.MAX_SOURCE_CHARS:,} characters" in bare
  assert "several defs" not in bare and mp.ONE_DEF in bare and "So may a call" not in bare


def test_the_calls_example_compiles_and_hands_over_no_survival_policy():
  """#434's rule for every worked example, on the served world and the
  practice's alike."""
  text = mp.CALLS
  example = textwrap.dedent(text[text.index("  def greet"):text.index("\n\nAn argument")])
  for facts in (HOME, st.hand_facts()):
    proc = compile_ok(example.strip("\n") + "\n", facts)
    assert proc.name == "greet" and set(proc.defs) == {"sway"}
  for word in ("charge", "battery", "station", "dock", "rack"):
    assert word not in example, word


def test_the_served_rule_says_the_librarys_calls_and_the_index_says_calls():
  rule = ov.hands_procedure_rule(HOME.places)
  assert ov.HANDS_PROCEDURE_CALLS in rule
  assert f"A source may hold several defs, all within its {lang.MAX_SOURCE_CHARS:,}" in rule
  assert mp.vocabulary_index(HOME)["calls"] is True
  assert mp.vocabulary_index(HOME)["callDepth"] == lang.MAX_CALL_DEPTH
  assert mp.vocabulary_index(NO_CALLS)["calls"] is False


def test_procedure_new_on_a_define_that_takes_arguments_says_why_it_ran_nothing():
  """#264's rule: what the robot asked for and did not get is said where
  it reads -- a def that takes arguments is called, never run."""
  from pluggybot.mind.overseer import PROCEDURE_NEW
  life = stub_life(WORLD)
  L = lib.Library(HOME)
  life.overseer = SimpleNamespace(library=L, event_map=None)
  decision = Decision(action=PROCEDURE_NEW, reason="r", define={
    "name": "turn", "source": "def turn(axis):\n  move(axis, 0.5)\n"})
  life.body.run(life._after_decision_routine(decision))
  assert not [e for e in life.errands if e.program is not None]
  assert ("ran nothing: `procedure:new` runs the procedure the same answer defines, and turn "
          "takes axis: it runs only when another procedure calls it"
          ) in life.thoughts.read("History.md")


# ---- the practice ---------------------------------------------------------------


def test_a_pass_is_kept_as_the_source_it_ran_its_callees_in_it():
  """The KEPT protocol keeps each pass whole: a callee changed is another
  program, and in practice a source calls its own defs alone -- a kept
  procedure is shown, never called -- so no later pass can change what an
  earlier one called."""
  from pluggybot.evaluation import practice as ep
  a = "def lid():\n  grip()\ndef grip():\n  wait(1)\n"
  b = "def lid():\n  grip()\ndef grip():\n  wait(2)\n"
  kept, _ = ep.keep([{"name": "lid", "source": a, "chest": 0, "attempt": 1},
                     {"name": "lid", "source": a, "chest": 1, "attempt": 2},
                     {"name": "lid", "source": b, "chest": 2, "attempt": 1}])
  assert [(k["source"], k["passed"]) for k in kept] == [(a, [[0, 1], [1, 2]])]
  [why] = refusals("def again():\n  lid()\n", st.hand_facts())
  assert why.startswith("line 2: unknown verb 'lid'") and why.endswith(
    "and no def of this source is named so"), why


def test_a_try_says_the_def_each_step_ran_in():
  life = _stub_life()
  life.body.go_to_routine = lambda *a, **kw: tick.result(False)
  r, _ = run("def a():\n  b()\ndef b():\n  wait(1)\n  drive_to(1, 1)\n", life)
  report = mp.run_report(1, r, {"ok": False, "reason": "x"}, 1.0, False)
  assert "and stopped at line 5, drive_to (in b): did not arrive" in report, report
  assert "  line 4 wait (in b): ok" in report, report


# ---- the review's (each reproduced before its fix) ------------------------------


def test_a_rewrite_that_would_call_round_through_its_caller_keeps_the_one_it_had():
  """`check` judged the define on the library as it stood, where `b`
  still called the old `a`: passed, the undefine took `a`, and the define
  was then refused -- the robot's procedure gone in the answer that was
  rewriting it (#264's loss). Judged on the library it makes, it is the
  loop it is, and nothing is taken."""
  L = lib.Library(HOME)
  L.define("a", "def a():\n  wait(1)\n")
  L.define("b", "def b():\n  a()\n")
  [why] = L.check("a", "def a():\n  b()\n", freeing="a")
  assert why == "b calls a, which calls b: a def may not call itself, by way of another or not"
  life = stub_life(WORLD)
  life.overseer = SimpleNamespace(library=L, event_map=None)
  life._define(Decision(action="idle", reason="r", undefine="a",
                        define={"name": "a", "source": "def a():\n  b()\n"}))
  assert L.names() == ("a", "b") and L.runnable() == ("a", "b")
  assert "could not rewrite the procedure a -- the one I had is kept" in life.thoughts.read(
    "History.md")


def test_an_undefine_beside_a_define_of_another_says_what_it_stopped_itself():
  life = stub_life(WORLD)
  L = lib.Library(HOME)
  life.overseer = SimpleNamespace(library=L, event_map=None)
  L.define("x", "def x():\n  wait(1)\n")
  L.define("b", "def b():\n  x()\n")
  L.define("c", "def c():\n  b()\n")
  life._define(Decision(action="idle", reason="r", undefine="x",
                        define={"name": "y", "source": "def y():\n  wait(2)\n"}))
  history = life.thoughts.read("History.md")
  assert ("forgot the procedure x -- b, which calls x, cannot run until a x is written again; "
          "c, which calls b, cannot run either") in history, history
  assert "wrote the procedure y\n" in history or history.rstrip().endswith("wrote the procedure y")


def test_a_name_passed_on_to_an_argument_its_callee_never_uses_keeps_its_kind():
  src = ('def m():\n  f("left")\ndef f(h):\n  g(h)\n  open_hand(h)\n'
         'def g(k):\n  wait(1)\n')
  proc = compile_ok(src)
  assert proc.defs["f"].names == {"h"}
  r, _ = run(src.replace("open_hand(h)", "wait(2)"))
  assert r["ok"] and [s["verb"] for s in r["steps"]] == ["wait", "wait"]


def test_a_call_of_one_that_cannot_run_is_no_procedure_that_calls_no_verb():
  L = lib.Library(HOME)
  L.define("b", "def b():\n  wait(1)\n")
  L.define("c", "def c():\n  b()\n")
  L.undefine("b")
  with pytest.raises(lib.LibraryRefused) as e:
    L.define("a", "def a():\n  c()\n")
  assert not any("calls no verb" in r for r in e.value.reasons), e.value.reasons


def test_a_kept_procedure_named_as_a_verb_still_runs_and_only_a_def_it_calls_may_not():
  """Its first def is run, never called by its own source: refused, a
  kept procedure named so before calls were would stop running at the
  deploy's load."""
  assert compile_ok("def wait():\n  face(1)\n").name == "wait"
  assert any("'face' is the language's own name" in r
             for r in refusals("def a():\n  wait(1)\ndef face():\n  wait(2)\n"))


def test_the_readout_names_each_procedures_own_defs_and_an_argument_it_set():
  L = lib.Library(HOME)
  L.define("pa", "def pa():\n  h()\ndef h():\n  x = 1\n  wait(1)\n")
  L.define("pb", "def pb():\n  h()\ndef h():\n  x = 2\n  wait(1)\n")
  L.define("top", "def top(z=1.0):\n  pa()\n  pb()\n  z += 1\n  q(3)\n"
                  "def q(n):\n  n = n * 2\n  wait(1)\n")
  r, _ = run_entry(L, "top")
  assert r["ok"] and r["locals"] == {"z": 2.0, "pa.h.x": 1.0, "pb.h.x": 2.0, "q.n": 6.0}
