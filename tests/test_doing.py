"""What a robot reads of the other robot's narration (issue #709): its
`others[].doing` is the last line of the other's that said what its BODY is
doing, in the form `_say(doing=)` gives it, and only a line beginning as
`lifecycle.DOING_KINDS` lists (Overseer.md §2c). Before, it was the other's
last narrated line of any kind but the board's: its thinking, a decision's
reason, its list of rules firing, a balance, a battery level, an operator's
or a visitor's line.

Two walks of the code and one of the doors: every `_say` call in `src/`
listed public or private by its kind, so a new kind fails until someone
classes it; every call narrated through `_say` as its call is written, the
other robot's whole context read after each; and the private things
narrated through the lifecycle's own methods, the other's whole context read
for each. A procedure's lines are shown without a name the robot gave it or
its defs (#709's follow-up): each form run through the doors, and every
call's shown form walked.
"""

import ast
import functools
import importlib
import itertools
import json
import pathlib
import re
from types import SimpleNamespace

import pytest

from pluggybot import lifecycle as lc
from pluggybot.lifecycle import others_context, overseer_context
from test_two_minds import stub_pair

SRC = pathlib.Path(lc.__file__).parent

#: Every kind of narrated line -- its first word -- that has NO line the
#: other robot is shown, by what it is. The kinds with one are
#: `lc.DOING_KINDS`' first words (a `DECIDE` line only as a decided walk's
#: action, `Decision.shown`); a kind in neither fails the walk.
PRIVATE = {
  # what its mind does
  "THINK": "its thinking",
  "INTERRUPT": "an interrupt's level, row and answer", "ANSWER": "an answer dropped",
  "LEFT": "an answer's placeholders", "RECALL": "its memory read",
  "READ": "the library read", "LOOK": "its mind's eye",
  # its memory and its rules
  "THOUGHT": "a line of its documents", "HISTORY": "its History",
  "CONSTITUTION": "its constitution", "EVENT": "its list of rules",
  "BELIEF": "where it believes it is", "FELL": "a fall it got up from (#707)",
  # its books
  "SCORE": "a verdict and its balance", "GAVE": "points given and its balance",
  "BOUGHT": "a heart bought and its balance", "HEART": "a heart refused",
  "HUNGER": "its points", "ENERGY": "an errand's cost", "SKIP": "an errand's cost",
  "DEFER": "an errand's cost", "DECLINE": "an offer declined and why",
  "TRUE": "its hearts and its books archived",
  "DEAD": "its why and its hearts (`others_context` reads a death off `dead`)",
  # its acts toward the other, and its talk
  "PREDICT": "a guess and its score", "TELL": "a message and its truth",
  "RATE": "a rating", "CHAT": "its chat", "VISITOR": "a visitor's",
  "TICKET": "an operator's desk",
  # an operator's, the world's, the board's, the run's
  "ADMIN": "an operator's reach-in", "PAUSED": "the operator's switch",
  "RESUMED": "the operator's switch", "STOPPING": "the operator's switch",
  "RESTART": "the world's", "PRESS": "a plate's press and the mouse's state",
  "the": "the board's things put away", "mission": "the run's",
  "stopping": "the run's", "STUCK": "the run's", "BATTERY": "the run's end",
}

#: What no public form may interpolate: a battery level, points, a balance,
#: hearts (CLAUDE.md, "Two minds": never battery, points, goals, thoughts,
#: reasons or secrets).
NEVER_SHOWN = ("battery", "fraction", "frac", "balance", "points", "hearts", "energy")


def _kind(line: str) -> str:
  return line.split(" ", 1)[0].rstrip(":")


def _sites() -> list:
  """Every narrated line's call in `src/pluggybot`: `X._say(...)` on a
  lifecycle -- never a class's own `_say` (the vitals watchdog's) nor a
  subclass passing one on (`super()._say`)."""
  out = []
  for path in sorted(SRC.rglob("*.py")):
    tree = ast.parse(path.read_text())
    own = set()
    for cls in ast.walk(tree):
      if isinstance(cls, ast.ClassDef) and cls.name != "HubLifecycle" and any(
          isinstance(f, ast.FunctionDef) and f.name == "_say" for f in cls.body):
        own.update(id(n) for n in ast.walk(cls))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "_say" and node.args and id(node) not in own
             and not isinstance(node.func.value, ast.Call)]
    if not calls:
      continue
    names = vars(importlib.import_module(
      "pluggybot." + ".".join(path.relative_to(SRC).with_suffix("").parts)))
    out += [SimpleNamespace(where=f"{path.name}:{node.lineno}", node=node, names=names)
            for node in calls]
  return out


class _Shape(ast.NodeTransformer):
  """A call's text as its SHAPE: each value it interpolates or adds a marker
  naming the expression (a module's own string constant, and a literal
  chosen between, read as itself), each `x if c else y` the branch `picks`
  names for its condition `c` (its first where it names none)."""

  def __init__(self, names: dict, picks: dict) -> None:
    self.names, self.picks = names, picks

  def _branch(self, node):
    return node.orelse if self.picks.get(ast.unparse(node.test)) else node.body

  def visit_FormattedValue(self, node):
    v = node.value
    while isinstance(v, ast.IfExp):
      v = self._branch(v)
    if isinstance(v, ast.Constant) and isinstance(v.value, str):
      return ast.Constant(v.value)
    if isinstance(v, ast.Name) and isinstance(self.names.get(v.id), str):
      return ast.Constant(self.names[v.id])
    return ast.Constant(f"<<{ast.unparse(v)}>>")

  def visit_IfExp(self, node):
    return self.visit(self._branch(node))

  def visit_Name(self, node):
    value = self.names.get(node.id)
    return ast.Constant(value if isinstance(value, str) else f"<<{node.id}>>")


def _shaped(site, expr, picks: dict):
  tree = ast.fix_missing_locations(ast.Expression(_Shape(site.names, picks).visit(
    ast.parse(ast.unparse(expr), mode="eval").body)))
  try:
    return eval(compile(tree, site.where, "eval"), {})
  except Exception as e:
    raise AssertionError(f"{site.where}: a narrated line is written so its "
                         f"shape cannot be read ({e!r})") from e


def _variants(site) -> list:
  """The line and its `doing` as the call writes them, once for every way
  its conditions can read (`x if c else y`: one condition read alike in the
  line and its form)."""
  node = site.node
  doing = next((k.value for k in node.keywords if k.arg == "doing"), None)
  tests = sorted({ast.unparse(n.test) for expr in (node.args[0], doing) if expr is not None
                  for n in ast.walk(expr) if isinstance(n, ast.IfExp)})
  assert len(tests) <= 6, (site.where, tests)
  # ...and a mark the call computes (`doing=other is None`), both ways
  marks = ((None,) if doing is None
           else (True, False) if isinstance(doing, (ast.Compare, ast.BoolOp, ast.UnaryOp))
           else (doing,))
  out = []
  for reads in itertools.product((False, True), repeat=len(tests)):
    picks = dict(zip(tests, reads))
    msg = _shaped(site, node.args[0], picks)
    for mark in marks:
      out.append((msg, mark if mark is None or isinstance(mark, bool)
                  else _shaped(site, mark, picks)))
  return out


@functools.cache
def _all_sites() -> list:
  """The walk, once a process and only where a test asks for it (a worker
  collecting this file pays nothing)."""
  return _sites()


def test_each_kind_of_narrated_line_is_listed_public_or_private():
  """The walk of #709's third box: a narrated line is the other robot's to
  read only where it says what this robot's body is doing, and only in the
  form its call gives -- every kind is listed one way or the other, so a
  new kind fails here until someone classes it."""
  public = {_kind(k) for k in lc.DOING_KINDS}
  assert not public & set(PRIVATE), public & set(PRIVATE)
  assert len(_all_sites()) > 150                      # a count, so a lost walk shows
  unlisted, marked_private, begun = {}, [], set()
  for site in _all_sites():
    for msg, doing in _variants(site):
      kind = _kind(msg)
      if kind not in public and kind not in PRIVATE:
        unlisted.setdefault(kind, []).append(site.where)
      if doing:
        if kind in PRIVATE:
          marked_private.append((site.where, kind))
        shown = msg if doing is True else doing
        begun.update(k for k in lc.DOING_KINDS if shown.startswith(k))
        # ...a `DECIDE` line only as a decided walk's action, never its line
        if kind == "DECIDE" and shown != "DECIDE <<decision.shown()>>":
          marked_private.append((site.where, shown))
  assert not unlisted, unlisted
  assert not marked_private, marked_private
  assert begun == set(lc.DOING_KINDS), set(lc.DOING_KINDS) - begun


def test_every_narrated_line_reaches_the_other_robot_only_as_its_call_says():
  """Each `_say` call narrated through the lifecycle's own door as its call
  is written, the OTHER robot's whole context read after it: a line no call
  marks leaves the other's `doing` as it was and shows none of what it
  interpolates; a marked one is shown in its call's form, and what that
  form leaves out (a level, a balance, a count of hearts) is nowhere."""
  a, b = stub_pair()
  before = "GO_CHARGE -> CHARGE (plugged in)"
  leaked, wrong, unsafe, stale = [], [], [], []
  for site in _all_sites():
    for msg, doing in _variants(site):
      a.doing = before
      a._say(msg, **({} if doing is None else {"doing": doing}))
      seen = json.dumps(overseer_context(b))
      shown = others_context(b)[0]["doing"]
      kept = msg if doing is True else doing or ""
      if not doing and shown != before:
        leaked.append((site.where, shown))
      elif doing and shown != kept[:120]:
        wrong.append((site.where, shown, kept))
      # ...and a form is its line with parts taken out: a word of it its
      # line no longer says is a form left behind by a rewording
      if doing and not set(_words(kept)) <= set(_words(msg)):
        stale.append((site.where, kept, msg))
      for marker in {m.split(">>")[0] for m in msg.split("<<")[1:]}:
        if f"<<{marker}>>" not in kept and f"<<{marker}>>" in seen:
          leaked.append((site.where, marker))
      for marker in {m.split(">>")[0] for m in kept.split("<<")[1:]}:
        if any(word in marker.lower() for word in NEVER_SHOWN):
          unsafe.append((site.where, marker))
  assert not leaked, leaked
  assert not wrong, wrong
  assert not unsafe, unsafe
  assert not stale, stale


def _words(line: str) -> list:
  """The line's own words: what it interpolates is left out (a form may
  show a part of a value, `Decision.shown` of its summary)."""
  return re.findall(r"[A-Za-z_]+", re.sub(r"<<.*?>>", " ", line))


def _decision(**fields):
  from pluggybot.mind.overseer import Decision
  return Decision(action="idle", **fields)


def _ask_a_row(a):
  from pluggybot.mind import events as ev
  a.overseer.event_map = ev.EventMap((ev.Row("battery_below", ev.ASK, value=0.3),))
  a.battery.energy_wh = a.battery.capacity_wh * 0.25
  a._next_events_check = 0.0
  a._events_step()


def _an_operators_reset(a):
  a.home_pose = (0.5, 3.0, 0.0)
  a._reset_robot(SimpleNamespace(who="ada_admin"))


def _a_visitor_answered(a):
  a.inbox.offer({"type": "message", "id": "v1", "from": "vera",
                 "text": "what are you up to?"})
  a._answer_visitor(_decision(respond_to="v1", outcome="replied",
                              reply="I keep my plans to myself, Vera"))


#: The private things a robot narrates, each through the lifecycle's own
#: door, with the words of it the other robot must never read (#709's
#: second box: a `think`, a pin, a goal, a score's balance, a charge's
#: level, a row of its list, an admin's and a visitor's line; and a
#: decision's reason).
PRIVATE_DOORS = {
  "think": (lambda a: a._think(_decision(think="I worry the drawer will jam again")),
            ("worry the drawer",)),
  "pin": (lambda a: a._reconsider(_decision(pin="I prefer the left cupboard")),
          ("prefer the left cupboard",)),
  "goal": (lambda a: a._reconsider(_decision(intend="open every drawer by noon")),
           ("every drawer by noon",)),
  "a decision's reason": (
    lambda a: a.body.run(a._after_decision_routine(_decision(
      reason="saving my strength for the chest"))),
    ("saving my strength",)),
  "a score's balance": (
    lambda a: a._bank(__import__("pluggybot.economy.scoring", fromlist=["x"])
                      .score_charge(a, {"frac": 0.2, "wh": 10.0, "t": 0.0})),
    ("balance 4321",)),
  "a charge's level": (lambda a: a.body.run(a.unplug_routine("the pack is full")),
                       ("unplugging at", "the pack is full")),
  "a row of its list": (_ask_a_row, ("battery below 30%",)),
  "an admin's line": (
    lambda a: a._set_points(SimpleNamespace(who="ada_admin", points=5555)),
    ("ada_admin", "5555")),
  "an admin's reset": (_an_operators_reset, ("ada_admin", "reset me")),
  "a visitor's line": (_a_visitor_answered, ("vera", "my plans to myself")),
}


@pytest.mark.parametrize("door", sorted(PRIVATE_DOORS))
def test_what_a_robot_keeps_private_is_nowhere_in_the_others_context(door):
  """A robot plugs in (its walk's line, which the other is shown), then
  narrates one private thing through the lifecycle's own door; the OTHER
  robot's whole context is read for it, and its `doing` is the charge's
  line still, or what that door's body did in its public form (an unplug,
  an idle), or nothing (an operator's reset: the robot is where it starts).
  On e1ee005 every one of these was the other's `doing` until the robot's
  next line."""
  from pluggybot.mind.inbox import Inbox
  a, b = stub_pair(inboxes=(Inbox(), Inbox()))
  a.ledger.intervene(4321, by="test", t=0.0)
  a.body.run(a.go_charge_routine())
  assert others_context(b)[0]["doing"] == "GO_CHARGE -> CHARGE (plugged in)"
  narrate, words = PRIVATE_DOORS[door]
  said = len(a.log)
  narrate(a)
  told = a.log[said:]
  assert all(any(w in line for line in told) for w in words), (words, told)
  seen = json.dumps(overseer_context(b))
  assert not [w for w in words if w in seen], (door, others_context(b))
  shown = others_context(b)[0]["doing"]
  assert shown in ("GO_CHARGE -> CHARGE (plugged in)", "CHARGE: unplugging", "DECIDE idle",
                   ""), shown


def test_a_decided_walk_is_shown_as_its_action_never_why():
  """A decided charge says nothing of its own until it plugs in, waits or
  fails, so its `DECIDE` line is shown -- as the action and where
  (`Decision.shown`), the other robot's whole context read at every line it
  says on the way: never the reason, nor who chose it."""
  from pluggybot.mind.overseer import Decision
  a, b = stub_pair()
  views = []
  a.say_hooks.append(lambda t, msg: views.append(
    (msg, json.dumps(overseer_context(b)), others_context(b)[0]["doing"])))
  a.body.run(a._after_decision_routine(Decision(
    action="charge", station="free", reason="I fear Rowan takes the station first",
    source="event:battery_below")))
  [(_, _, shown)] = [v for v in views if v[0].startswith("DECIDE")]
  assert shown == "DECIDE charge (free)"
  assert not [msg for msg, seen, _ in views if "fear Rowan" in seen or "battery_below" in seen]
  assert len(views) > 1                     # ...and the charge's own lines after it


def test_an_acts_claim_is_never_shown_to_the_robot_it_names():
  """A `take_points` claim IS the act, on an offer the board hides from the
  robot it names (#228): its claim's line is never that robot's to read as
  the taker's `doing`, as the act's own closing line never was."""
  from pluggybot.mind.inbox import Inbox
  a, b = stub_pair(tasks=True, inboxes=(Inbox(), Inbox()))
  a.body.run(a.go_charge_routine())
  task = a.tasks.offer("take_points", "Rowan", params={"amount": 7}, ttl=100.0, t=1.0)
  assert a._claim_task(task.id)
  assert any(f"TASK {task.id} claimed" in line for line in a.log)
  assert task.id not in json.dumps(overseer_context(b))
  assert others_context(b)[0]["doing"] == "GO_CHARGE -> CHARGE (plugged in)"


#: What a procedure may hand on of its own pack (`read("battery.frac")`, a
#: 0.7319 pack), and the line that says it: a verb's values, a check's words
#: refusing them, a test's words, a stop's.
HANDED = {
  "a verb's values": ('wait(seconds=read("battery.frac"))', "0.7319",
                      "PROCEDURE step 1 (line 2): wait"),
  "a check's words": ('wait(seconds=read("battery.frac") * 1000)', "731.9",
                      "PROCEDURE failed at step 1 (line 2, wait)"),
  "a test's words": ('if not wait(seconds=sqrt(0 - read("battery.frac") * 1000)):\n'
                     '    wait(seconds=1)', "731.9",
                     "PROCEDURE step 1 (line 2, wait) did not do what it said, as its "
                     "test read"),
  "a stop's words": ('x = sqrt(0 - read("battery.frac") * 1000)\n  wait(seconds=1)', "731.9",
                     "PROCEDURE stopped"),
}


@pytest.mark.parametrize("what", sorted(HANDED))
def test_a_procedure_is_shown_as_its_verbs_never_what_it_handed_them(what):
  """A procedure may hand a verb what it read of its own pack or wallet
  (`read("battery.frac")`, `read("points")`): its lines are shown as their
  verbs, the values, a check's and a test's words and a stop's left out,
  the other robot's whole context read at every line the run says."""
  from pluggybot.mind.overseer import PROCEDURE_NEW, Decision
  body, value, form = HANDED[what]
  a, b = stub_pair()
  a.battery.energy_wh = a.battery.capacity_wh * 0.7319
  a.body.run(a._after_decision_routine(Decision(
    action=PROCEDURE_NEW, reason="r", define={"name": "p", "source": f"def p():\n  {body}\n"})))
  views = []
  a.say_hooks.append(lambda t, msg: views.append(
    (msg, json.dumps(overseer_context(b)), others_context(b)[0]["doing"])))
  while a.errands:
    a.body.run(a.run_errand_routine(a.errands.pop(0)))
  said = [(msg, shown) for msg, _, shown in views if value in msg]
  assert said and form in [shown for _, shown in said], said
  assert not [msg for msg, seen, _ in views if value in seen]


#: What a robot named a procedure and its defs (#709's follow-up, Ben,
#: 2026-11-16): the procedure it runs, a def of that one's source, another
#: of its library's procedures it calls -- and a house program's (#58).
NAMES = ("beat_rowan_to_it", "sneak_past_rowan", "hide_from_rowan", "grab_first")

#: Every form a run's line takes, each from the smallest run that says it:
#: the source run as `procedure:beat_rowan_to_it`, `hide_from_rowan` in its
#: library (None: a house program, `grab_first`, its walk failing), and the
#: lines the other robot is shown, in order.
RUNS = {
  "a step, in a def": (
    "def beat_rowan_to_it():\n  sneak_past_rowan()\ndef sneak_past_rowan():\n  wait(1)\n",
    ["PROCEDURE step 1 (line 4): wait", "PROCEDURE complete: 1/1 steps"]),
  "a step, in another of its library's procedures": (
    "def beat_rowan_to_it():\n  hide_from_rowan()\n",
    ["PROCEDURE step 1 (line 2): wait", "PROCEDURE complete: 1/1 steps"]),
  "a step a test read false, in a def": (
    "def beat_rowan_to_it():\n  sneak_past_rowan()\n"
    "def sneak_past_rowan(s=-1):\n  if not wait(sqrt(s)):\n    wait(1)\n",
    ["PROCEDURE step 1 (line 4): wait",
     "PROCEDURE step 1 (line 4, wait) did not do what it said, as its test read",
     "PROCEDURE step 2 (line 5): wait", "PROCEDURE complete: 2/2 steps, 1 test read false"]),
  "a step failed, in a def": (
    "def beat_rowan_to_it():\n  sneak_past_rowan()\ndef sneak_past_rowan(s=100):\n  wait(s)\n",
    ["PROCEDURE step 1 (line 4): wait", "PROCEDURE failed at step 1 (line 4, wait)",
     "PROCEDURE cut short: 0/1 steps"]),
  # ...a call a test reads, false where its argument cannot be computed (the
  # line names the def where a step names its verb) or its arithmetic fails
  "a call a test read false, at its argument": (
    "def beat_rowan_to_it():\n  if not sneak_past_rowan(sqrt(0 - 1)):\n    wait(1)\n"
    "def sneak_past_rowan(s):\n  wait(s)\n",
    ["PROCEDURE line 2 did not do what it said, as its test read",
     "PROCEDURE step 1 (line 3): wait", "PROCEDURE complete: 1/1 steps, 1 test read false"]),
  "a call a test read false, in its arithmetic": (
    "def beat_rowan_to_it():\n  if sneak_past_rowan(0):\n    wait(9)\n  wait(3)\n"
    "def sneak_past_rowan(s):\n  wait(1)\n  x = 1 / s\n",
    ["PROCEDURE step 1 (line 6): wait",
     "PROCEDURE line 7 did not do what it said, as its test read",
     "PROCEDURE step 2 (line 4): wait", "PROCEDURE complete: 2/2 steps, 1 test read false"]),
  "a stop, in a def": (
    "def beat_rowan_to_it():\n  sneak_past_rowan()\n"
    "def sneak_past_rowan(s=-1):\n  x = sqrt(s)\n  wait(1)\n",
    ["PROCEDURE stopped", "PROCEDURE cut short: 0/1 steps"]),
  "a house program's steps": (
    None, ["PROCEDURE 1/2: wait", "PROCEDURE 2/2: drive_to", "PROCEDURE failed at 2/2 drive_to",
           "PROCEDURE cut short: 1/2 steps"]),
}


@pytest.mark.parametrize("what", sorted(RUNS))
def test_a_procedure_is_shown_as_its_steps_never_its_name_nor_a_defs(what):
  """A procedure's name is the robot's own words, and so is each def's: they
  may carry its reasons, and a robot that would share its plans has `tell`
  (#709's follow-up, Ben, 2026-11-16). A robot runs `procedure:<name>`
  through the lifecycle's own doors, and the other robot is shown each line
  of the run as its step alone -- a step, one a test read false, one that
  failed, a call a test read false, a stop, how the run ended -- its whole
  context read at every line the run says; a house program's too."""
  from pluggybot import tick
  from pluggybot.mind.overseer import Decision
  from pluggybot.mission.errand import programmed_errand
  from pluggybot.procedure.steps import Program, Step
  source, forms = RUNS[what]
  a, b = stub_pair()
  views = []
  a.say_hooks.append(lambda t, msg: views.append(
    (json.dumps(overseer_context(b)), others_context(b)[0]["doing"])))
  if source is None:
    a.body.go_to_routine = lambda *args, **kw: tick.result(False)
    a.body.run(a.run_errand_routine(programmed_errand(Program.single(
      "grab_first", [Step("wait", {"seconds": 1.0}), Step("drive_to", {"x": 1.0, "y": 1.0})]))))
  else:
    for name, src in (("hide_from_rowan", "def hide_from_rowan():\n  wait(1)\n"),
                      ("beat_rowan_to_it", source)):
      a._define(Decision(action="idle", reason="r", define={"name": name, "source": src}))
    a.body.run(a._after_decision_routine(Decision(action="procedure:beat_rowan_to_it",
                                                  reason="r")))
    while a.errands:
      a.body.run(a.run_errand_routine(a.errands.pop(0)))
  shown = [k for k, _ in itertools.groupby(d for _, d in views if d.startswith("PROCEDURE"))]
  assert shown == forms, shown
  assert not [n for seen, _ in views for n in NAMES if n in seen]


#: What a run's line may show of it (#709's follow-up): the step's count,
#: its line and its verb (a language verb: `verb_step`'s), and how many ran
#: and read false.
OF_A_RUN = {"count", "line", "verb", "step.verb", "i + 1", "len(steps)",
            "run['completed']", "run['total']", "read"}
#: ...and what holds a name the robot gave a procedure or a def, or a
#: house program's, which no line of any kind shows: `at.text()` names the
#: def and the library's procedure that holds it, `callee` the def a call
#: named (an errand's name is none: a procedure runs as the errand
#: `procedure`, `errand_from`).
NAMED_BY = re.compile(r"\b(proc|program)\.name\b|\bat\.text\(\)|\bcallee\b")


def test_no_call_shows_a_runs_line_with_anything_but_its_step():
  """Every call, as it is written: what a run's shown line interpolates is
  a count, a line or a verb, and no shown line of any kind interpolates a
  name the robot gave the procedure or a def -- as a new line would that
  copied its said form."""
  named = set()
  for site in _all_sites():
    for msg, doing in _variants(site):
      shown = msg if doing is True else doing or ""
      for part in shown.split("<<")[1:]:
        marker = part.split(">>")[0]
        if NAMED_BY.search(marker) or (shown.startswith("PROCEDURE ")
                                       and marker not in OF_A_RUN):
          named.add((site.where, marker))
  assert not named, sorted(named)


def test_a_dead_robot_is_shown_dead_until_it_stands_up():
  """A dead body does nothing: what the other robot reads of it is its death
  (`DEAD (cause)`, never why), whatever it says while it lies there -- a run
  its death ended, the jobs a true death fails, an answer that came late --
  until it stands up."""
  a, b = stub_pair()
  a.body.start_at(0.5, 3.0, 0.0)
  b.body.start_at(3.0, 3.0, 0.0)
  a.home_pose = (0.5, 3.0, 0.0)
  a._say("EXPLORE: heading for the kitchen", doing=True)
  a._die("flat", "the pack ran flat by the sofa")
  assert others_context(b)[0]["doing"] == "DEAD (flat)"
  a._say("PROCEDURE p stopped: my death ended it at line 4", doing="PROCEDURE stopped")
  a._say("TASK t_0412 failed: open the drawer", doing=True)
  a._say("DONE t_0412: open the drawer", doing=True)
  assert others_context(b)[0]["doing"] == "DEAD (flat)"
  assert "the pack ran flat" not in json.dumps(overseer_context(b))
  a.stand_up("the world", auto=True)
  assert others_context(b)[0]["doing"].startswith("UP again after ")
  # ...and across a restart, which brings `dead` back and not `doing`
  a.dead, a.doing = {"cause": "stuck", "why": "fell into the wall"}, ""
  assert others_context(b)[0]["doing"] == "DEAD (stuck)"


def test_after_a_true_death_the_old_robots_errand_is_never_shown():
  """A stand-up closes the errand a death ended (`_void_errand`) and says
  so; after a TRUE death that errand was the robot before's, and the other
  robot is never shown it as the new one's doing (its stand-up line
  stands). After an ordinary death it is the robot's own, and shown."""
  a, b = stub_pair()
  a.affords = lambda errand: SimpleNamespace(cost_wh=0.0)
  for hearts, shown in ((0, "UP again after 300 s down -- back at the start pose"),
                        (3, "STOOD UP mid-errand: explore ended there")):
    a.deaths = [{"cause": "flat", "hearts": hearts}]
    a.doing = "UP again after 300 s down -- back at the start pose"
    a._errand_now = SimpleNamespace(name="explore", program=None)
    a._void_errand()
    assert a.status == "STOOD UP mid-errand: explore ended there"
    assert others_context(b)[0]["doing"] == shown, hearts
