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
for each.
"""

import ast
import importlib
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
#: action, `lc.DECIDED_SHOWN`); a kind in neither fails the walk.
PRIVATE = {
  # what its mind does
  "THINK": "its thinking",
  "INTERRUPT": "an interrupt's level, row and answer", "ANSWER": "an answer dropped",
  "LEFT": "an answer's placeholders", "RECALL": "its memory read",
  "READ": "the library read", "LOOK": "its mind's eye",
  # its memory and its rules
  "THOUGHT": "a line of its documents", "HISTORY": "its History",
  "CONSTITUTION": "its constitution", "EVENT": "its list of rules",
  "BELIEF": "where it believes it is",
  # its books
  "SCORE": "a verdict and its balance", "GAVE": "points given and its balance",
  "BOUGHT": "a heart bought and its balance", "HEART": "a heart refused",
  "HUNGER": "its points", "ENERGY": "an errand's cost", "SKIP": "an errand's cost",
  "DEFER": "an errand's cost", "DECLINE": "an offer declined and why",
  "TRUE": "its hearts and its books archived",
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
  chosen between, read as itself), each `x if c else y` the branch `pick`
  names."""

  def __init__(self, names: dict, pick: int) -> None:
    self.names, self.pick = names, pick

  def visit_FormattedValue(self, node):
    v = node.value
    while isinstance(v, ast.IfExp):
      v = v.body if self.pick == 0 else v.orelse
    if isinstance(v, ast.Constant) and isinstance(v.value, str):
      return ast.Constant(v.value)
    if isinstance(v, ast.Name) and isinstance(self.names.get(v.id), str):
      return ast.Constant(self.names[v.id])
    return ast.Constant(f"<<{ast.unparse(v)}>>")

  def visit_IfExp(self, node):
    return self.visit(node.body if self.pick == 0 else node.orelse)

  def visit_Name(self, node):
    value = self.names.get(node.id)
    return ast.Constant(value if isinstance(value, str) else f"<<{node.id}>>")


def _shaped(site, expr, pick: int):
  tree = ast.fix_missing_locations(ast.Expression(_Shape(site.names, pick).visit(
    ast.parse(ast.unparse(expr), mode="eval").body)))
  try:
    return eval(compile(tree, site.where, "eval"), {})
  except Exception as e:
    raise AssertionError(f"{site.where}: a narrated line is written so its "
                         f"shape cannot be read ({e!r})") from e


def _variants(site) -> list:
  """The line and its `doing` as the call writes them, once a branch of
  each `if`/`else` it chooses between (an interpolation's own aside)."""
  node = site.node
  doing = next((k.value for k in node.keywords if k.arg == "doing"), None)
  branches = any(isinstance(n, ast.IfExp) for n in ast.walk(node.args[0])) or (
    doing is not None and any(isinstance(n, ast.IfExp) for n in ast.walk(doing)))
  out = []
  for pick in ((0, 1) if branches else (0,)):
    msg = _shaped(site, node.args[0], pick)
    out.append((msg, None if doing is None else _shaped(site, doing, pick)))
  return out


SITES = _sites()


def test_each_kind_of_narrated_line_is_listed_public_or_private():
  """The walk of #709's third box: a narrated line is the other robot's to
  read only where it says what this robot's body is doing, and only in the
  form its call gives -- every kind is listed one way or the other, so a
  new kind fails here until someone classes it."""
  public = {_kind(k) for k in lc.DOING_KINDS}
  assert not public & set(PRIVATE), public & set(PRIVATE)
  assert len(SITES) > 150                       # a count, so a lost walk shows
  unlisted, marked_private, begun = {}, [], set()
  for site in SITES:
    for msg, doing in _variants(site):
      kind = _kind(msg)
      if kind not in public and kind not in PRIVATE:
        unlisted.setdefault(kind, []).append(site.where)
      if doing:
        if kind in PRIVATE:
          marked_private.append((site.where, kind))
        shown = msg if doing is True else doing
        begun.update(k for k in lc.DOING_KINDS if shown.startswith(k))
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
  for site in SITES:
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
  show a part of a value, `Decision.what` of its summary)."""
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
  "a visitor's line": (_a_visitor_answered, ("vera", "my plans to myself")),
}


@pytest.mark.parametrize("door", sorted(PRIVATE_DOORS))
def test_what_a_robot_keeps_private_is_nowhere_in_the_others_context(door):
  """A robot plugs in (its walk's line, which the other is shown), then
  narrates one private thing through the lifecycle's own door; the OTHER
  robot's whole context is read for it, and its `doing` is still the
  charge's line. On e1ee005 every one of these was the other's `doing`
  until the robot's next line."""
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
  assert shown in ("GO_CHARGE -> CHARGE (plugged in)", "CHARGE: unplugging"), shown


def test_a_decided_walk_is_shown_as_its_action_never_why():
  """A decided charge says nothing of its own until it plugs in, waits or
  fails, so its `DECIDE` line is shown -- as the action and where
  (`Decision.what`), the other robot's whole context read at every line it
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


def test_a_step_is_shown_as_its_verb_never_the_values_it_was_handed():
  """A procedure may hand a verb what it read of its own pack or wallet
  (`read("battery.frac")`, `read("points")`): a step's line is shown as its
  verb, its values left out, the other robot's whole context read at every
  line the run says."""
  from pluggybot.mind.overseer import PROCEDURE_NEW, Decision
  a, b = stub_pair()
  a.battery.energy_wh = a.battery.capacity_wh * 0.7319
  a.body.run(a._after_decision_routine(Decision(
    action=PROCEDURE_NEW, reason="r", define={
      "name": "p", "source": 'def p():\n  wait(seconds=read("battery.frac"))\n'})))
  views = []
  a.say_hooks.append(lambda t, msg: views.append(
    (msg, json.dumps(overseer_context(b)), others_context(b)[0]["doing"])))
  while a.errands:
    a.body.run(a.run_errand_routine(a.errands.pop(0)))
  assert ("PROCEDURE p step 1 (line 2): wait(seconds=0.7319)", ) == tuple(
    msg for msg, _, _ in views if "step 1" in msg)
  assert [shown for msg, _, shown in views if "step 1" in msg] == [
    "PROCEDURE p step 1 (line 2): wait"]
  assert not [msg for msg, seen, _ in views if "0.7319" in seen]
