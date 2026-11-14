"""The procedure language: conditionals, bounded loops and arithmetic over
the step vocabulary (issue #166, rung two of agent-written code).

Python-SHAPED, never Python-RUN. The surface syntax is a `def` with a body,
because a model writes that far better than a bespoke dialect and a garbled
answer then becomes a measurement instead of a tax -- but the text is
PARSED (with `ast`) into this module's own small tree and interpreted by a
routine; nothing here calls `exec`, `eval` or `compile`, and a program that
uses any construct outside the grammar is refused with the line it broke,
before a single step runs. The grammar:

    def reach_and_look():                   # one def, no arguments
        budget(steps=60, seconds=300)       # optional, first; capped by code
        n = 0                               # locals, arithmetic
        drive_to(1.0, 2.0)                  # a verb, arguments literal or computed
        for i in range(3):                  # a literal count, <= MAX_ITER
            face(0.5 * i)
            if read("battery.frac") < 0.5:
                wait(2)
        while read("left.z") < 1.0 and n < 10:        # capped at MAX_ITER
            reach("left", 0.3, 0.2, read("left.z") + 0.05)
            n = n + 1
        withdraw("left")

  statements   verb(...)  name = expr  name += expr  if/elif/else  for name in
               range(N)  while cond  pass  return  budget(...)
  expressions  numbers, True/False, locals, read("sensor"), + - * / unary -,
               comparisons, and/or/not, parentheses
  nothing else no strings outside a verb's or read's argument, no attribute
               access, no calls but verbs and read, no imports, no I/O

Where the world's facts allow CALLS (`WorldFacts.calls`, #560), a source
holds one or more defs: the first is the procedure and runs, and a def calls
another by its name -- a def of its own source, or in a library another
procedure of the library -- as a statement or an `if`'s test, its arguments
numbers or names (a name is a verb's or read's, typed by its use, and names
join with +). One budget for the whole tree (the first def's), calls at most
`MAX_CALL_DEPTH` deep and never round to themselves, every statement counted
(`MAX_STATEMENTS`).

Totality by construction: every loop is bounded (a literal count, or the
cap), a program declares a step budget and a sim-time budget and the
vocabulary caps both, and the interpreter checks them at every verb -- which
is a safe point (steps.py).

Where it stands on the fence: a procedure reaches the motors only through a
verb (`move` is `Body.ramp_routine`), the scoring path not at all -- a
`Verdict` is sealed -- and nothing here writes `data.ctrl`.
"""

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Any

from pluggybot.finite import finite
from pluggybot.procedure import steps as st
from pluggybot.procedure.steps import Refused, WorldFacts
from pluggybot.tick import Routine

#: Caps, by code, on what a procedure may declare.
MAX_ITER = 100            # iterations of one loop (a `while` that reaches it fails)
MAX_STEPS = 200           # verb executions in one run; the default budget is lower
DEFAULT_STEPS = 60
MAX_BUDGET_S = st.MAX_BUDGET_S
DEFAULT_BUDGET_S = st.DEFAULT_BUDGET_S
#: A source's characters, every def in it: each kept one rides the served
#: mind's every turn (`text.MAX_PROCEDURES` of them). 3,000 since #560 (Ben,
#: 2026-11-04): of 4,838 sources minds wrote in practice, never told the
#: cap, 421 were over 1,500 and 47 over 3,000.
MAX_SOURCE_CHARS = 3000
#: ...and past this one is refused on its length alone, unparsed: under it,
#: a source over the cap is read and refused with every reason, where the
#: cap's alone hid them all (#615's smoke: three repairs of 5,458 characters
#: taught a mind its length and none of its six other faults). ⚠ A served
#: answer's source is cut one past the cap (`overseer`'s `define`), and so
#: hears the cap alone.
MAX_PARSE_CHARS = 65536
MAX_NESTING = 4
#: How deep calls go below the def that runs (#560): `a` calling `b` is one.
MAX_CALL_DEPTH = 4
#: Statements one run executes, its whole call tree (#560), about 0.1 s of
#: the physics thread at the 1 us a statement measured on the dev box:
#: every loop is bounded, but four nested in one def could run 10^8 (some
#: 100 s, both robots still), and loops in calls in loops multiply that.
MAX_STATEMENTS = 100000
#: A run's `stopped` where its code says it in words.
STOPS = {"statements": f"it ran {MAX_STATEMENTS:,} statements, the most one run may"}
#: How deep an expression nests. Each level was one frame of the parser's
#: recursion, and ~980 unary minuses (a source of 1,000 characters) raised
#: RecursionError out of the define and ended the served process (#500).
MAX_EXPR_DEPTH = 50
#: A name a procedure gives -- a variable's, a def's, an argument's -- is at
#: most this long, and the text says its rule (`name_rule`; #640: unsaid, 5
#: of 25 first table answers named a variable in capitals).
MAX_NAME_CHARS = 32
NAME_RE = re.compile(rf"^[a-z][a-z0-9_]{{0,{MAX_NAME_CHARS - 1}}}$")
#: Names a local may not shadow: the language's own words and every verb.
RESERVED = frozenset({"read", "budget", "range", "def", "and", "or", "not",
                      "if", "elif", "else", "for", "while", "in", "return",
                      "pass", "True", "False", "None"})


def called_only(proc) -> str:
  """What a procedure whose first def needs arguments is told, after its
  name: it runs only when another calls it -- and, its source holding
  several defs, that the first is the one that runs (#640: such sources
  most often put a helper first, as a Python script does)."""
  takes = ", ".join(p for p in proc.params if p not in proc.defaults)
  return (f"takes {takes}: it runs only when another procedure calls it"
          + (f"; of its defs the first, {proc.name}, is the one that runs, so put the one "
             "you mean to run first" if proc.defs else ""))


def name_rule() -> str:
  """What a name is, as the text says it and a refusal gives it."""
  return ("lowercase letters, digits and underscores, a letter first, at most "
          f"{MAX_NAME_CHARS} characters")


def name_fault(name: str, verbs=(), own=(), math: bool = False) -> str:
  """Why `name` is no name a procedure may give, or "": ⚠ AS TRUE AS THE
  CHECK IT ANSWERS (#640: a capitalised local was refused as "reserved"),
  each site asking what it refuses -- a local may not be a verb's or a
  def's, an argument not a maths function's either."""
  if not NAME_RE.match(name):
    return name_rule()
  if name in RESERVED:
    return "it is one of the language's own words"
  if name in verbs:
    return "it is a verb's name"
  if name in own:
    return "it is a def's name"
  if math and name in MATH:
    return "it is a maths function's name"
  return ""


_BIN = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
#: The functions an expression may call where the world's facts allow it
#: (`WorldFacts.math`, #493: an arc, a pivot's angle and a distance need
#: them), and how many numbers each takes. Angles in radians.
MATH = {"sin": 1, "cos": 1, "atan2": 2, "sqrt": 1}
#: ...and how one it lacks is named in its refusal.
_OP_TEXT = {ast.Mod: "%", ast.Pow: "**", ast.FloorDiv: "//", ast.MatMult: "@",
            ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^", ast.LShift: "<<",
            ast.RShift: ">>"}
_CMP = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
        ast.Eq: "==", ast.NotEq: "!="}


def _dotted_refusal(node: ast.Attribute) -> str:
  """A name with a dot, refused with what it most likely meant: a sensor
  written as Python, `spot.nx` for `read("spot.nx")` -- the commonest
  refusal of #493's flight, 37 times, while it said only that an
  Attribute is not part of this language."""
  from pluggybot.procedure.axes import SENSORS
  text = ast.unparse(node)
  if text in SENSORS:
    return f'{text} is a sensor: read it with read("{text}")'
  return (f'{text}: a name with a dot is not part of this language (a sensor '
          'is read with read("<sensor>"))')
#: What an expression the parser refused stands in as: no value, so the
#: world's check never refuses it again (a stand-in 0 read `tag 0 is no
#: cube`, a tag the source never wrote, #463). It never runs: a parse with
#: a reason is refused whole.
_REFUSED = ("refused",)


@dataclass(frozen=True)
class Def:
  """One def of a source past its first (#560): its arguments in order,
  their defaults -- ("num", v) or ("str", s) -- which of them hold names
  and which numbers (an argument it never uses holds either), and its
  body."""

  name: str
  params: tuple = ()
  defaults: dict = field(default_factory=dict)
  names: frozenset = frozenset()
  body: tuple = ()
  numbers: frozenset = frozenset()


@dataclass(frozen=True)
class Procedure:
  """A compiled procedure: its source (what the robot wrote, what rides the
  wire and the library) and the tree the interpreter walks -- its FIRST
  def's, with that def's arguments, the defs after it (`defs`), and the
  library's procedures it calls, compiled (`linked`, #560)."""

  name: str
  source: str
  body: tuple = ()
  steps_budget: int = DEFAULT_STEPS
  budget_s: float = DEFAULT_BUDGET_S
  #: how many verb calls the body contains statically -- a rough size, each
  #: call counting its callee's
  verbs: int = 0
  roles: dict = field(default_factory=dict)
  params: tuple = ()
  defaults: dict = field(default_factory=dict)
  names: frozenset = frozenset()
  defs: dict = field(default_factory=dict)
  linked: dict = field(default_factory=dict)
  #: the deepest chain of calls below it
  depth: int = 0
  numbers: frozenset = frozenset()

  @property
  def alone(self) -> bool:
    """Whether it runs with nothing passing it arguments: every one it
    takes has a default."""
    return all(p in self.defaults for p in self.params)

  def steps(self, role: str = st.DEFAULT_ROLE) -> tuple:
    """For the errand plumbing that sizes a program: the verb calls."""
    return tuple(range(self.verbs))

  def first(self, verb: str, arg: str, role: str = st.DEFAULT_ROLE):
    """The first LITERAL `arg` a `verb` call names anywhere in the body, in
    source order, or None -- `Program.first`'s shape."""
    def walk(block):
      for s in block:
        if s[0] == "verb":
          if s[1] == verb and arg in s[2] and s[2][arg][0] == "str":
            return s[2][arg][1]
        elif s[0] == "if":
          for cond, body in s[1]:
            found = walk(((cond[2],) if cond[0] == "try" else ()) + tuple(body))
            if found is not None:
              return found
          found = walk(s[2])
          if found is not None:
            return found
        elif s[0] in ("repeat", "while"):
          found = walk(s[3] if s[0] == "repeat" else s[2])
          if found is not None:
            return found
      return None
    return walk(self.body)

  def as_dict(self) -> dict:
    return {"name": self.name, "source": self.source,
            "budgetS": self.budget_s, "stepsBudget": self.steps_budget}

  @classmethod
  def from_dict(cls, spec: dict, facts: WorldFacts | None = None) -> "Procedure":
    return parse(str(spec.get("source", "")))


# ---- parsing: surface syntax -> the tree, or Refused ----------------------------


class _Parser:
  def __init__(self, source: str, verbs: dict | None = None,
               verb_tests: bool = False, math: bool = False, calls: bool = False,
               library: dict | None = None) -> None:
    self.source = source
    #: whether a verb may stand alone as an `if`'s test (`WorldFacts.verb_tests`)
    self.verb_tests = verb_tests
    #: whether an expression may call `MATH`'s functions (`WorldFacts.math`)
    self.math = math
    #: whether a source may hold several defs and a def call another
    #: (`WorldFacts.calls`, #560)...
    self.calls = calls
    #: ...and the library's procedures one may call, by name: compiled, or
    #: the reasons one is not (`Library`); None outside a library
    self.library = library
    #: The verbs it may call: `steps.VERBS`, and a world's own where it
    #: names them (`steps.language`, issue #493's hands).
    self.verbs = st.VERBS if verbs is None else verbs
    self.reasons: list[str] = []
    #: the call being parsed and its line, which every refusal inside it
    #: names, once: `line 3, wait: ...`, as History says a step that failed
    #: (#463). The CALL's line, never a part's: a call written across lines
    #: read as two calls of one verb.
    self.call: tuple[str, int] | None = None
    self.depth = 0
    #: the source's defs, by name: (arguments, defaults, line)
    self.own: dict[str, tuple] = {}
    #: the def being read, its arguments, how each was used -- as a name or
    #: a number, at its first line (`uses`) -- and which it sets; and every
    #: call a def makes, read once each callee's arguments are known (`sites`)
    self.fn = ""
    self.params: tuple = ()
    self.uses: dict[str, dict[str, dict]] = {}
    self.assigned: dict[str, dict[str, int]] = {}
    self.sites: list[tuple] = []
    #: whether a call named nothing it may call: its procedure's verbs are
    #: then not all counted
    self.unknown = False
    #: the names refused so far, each one's reason said at its first refusal
    self.named_bad: set[str] = set()

  def bad_name(self, node, name: str, why: str) -> None:
    """A name refused: why, the first time (every later use of an uppercase
    local repeating the rule ran a refusal list to dozens of lines)."""
    said = name in self.named_bad
    self.named_bad.add(name)
    self.refuse(node, f"{name!r} is not a name this language allows"
                      + ("" if said else f" ({why})"))

  def callable(self, name: str) -> bool:
    """A name a call may name: a def of this source, or the library's."""
    return self.calls and (name in self.own
                           or (self.library is not None and name in self.library))

  def use(self, param: str, kind: str, node) -> None:
    """An argument of the def being read, used as a `name` or a `number`."""
    self.uses.setdefault(self.fn, {}).setdefault(param, {}).setdefault(
      kind, getattr(node, "lineno", 0))

  def name(self, node, what: str) -> Any:
    """A name where a verb or `read` takes one: in quotes -- or, where defs
    take arguments, an argument that holds one, and names joined with +
    (#560: `read(hand + ".z")`)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
      return ("str", node.value)
    if self.calls and isinstance(node, ast.Name) and node.id in self.params:
      self.use(node.id, "name", node)
      return ("param", node.id)
    if self.calls and isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
      parts = (self.name(node.left, what), self.name(node.right, what))
      return _REFUSED if _REFUSED in parts else ("join", *parts)
    self.refuse(node, f"{what} takes a name in quotes" + (
      f", or an argument that holds one: {node.id} is no argument of {self.fn}'s"
      if self.calls and isinstance(node, ast.Name) else ""))
    return _REFUSED

  def refuse(self, node, what: str) -> None:
    if self.call:
      verb, line = self.call
      self.reasons.append(f"line {line}, {verb}: {what}")
    else:
      self.reasons.append(f"line {getattr(node, 'lineno', 0)}: {what}")

  # -- expressions --
  def expr(self, node) -> Any:
    self.depth += 1
    try:
      if self.depth > MAX_EXPR_DEPTH:
        self.refuse(node, f"an expression nests deeper than {MAX_EXPR_DEPTH}")
        return _REFUSED
      return self._expr(node)
    finally:
      self.depth -= 1

  def _expr(self, node) -> Any:
    if isinstance(node, ast.Constant):
      if isinstance(node.value, bool):
        return ("num", 1.0 if node.value else 0.0)
      if finite(node.value):
        return ("num", float(node.value))
      if isinstance(node.value, (int, float)):
        # past a float's range (309 digits, or 1e400): `float()` raised
        # OverflowError out of the define, and the served process ended (#500)
        self.refuse(node, "a number is too large to hold")
        return _REFUSED
      self.refuse(node, f"a {type(node.value).__name__} is not a number")
      return _REFUSED
    if isinstance(node, ast.Name):
      why = name_fault(node.id, self.verbs, self.own if self.calls else ())
      if why:
        self.bad_name(node, node.id, why)
      elif node.id in self.params:
        self.use(node.id, "number", node)
      return ("var", node.id)
    if isinstance(node, ast.Call):
      fn = node.func
      if isinstance(fn, ast.Name) and fn.id == "read":
        line = getattr(node, "lineno", 0)
        if (len(node.args) == 1 and not node.keywords and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)):
          # its line: a read outside a call is refused on it (`validate`)
          return ("read", node.args[0].value, line)
        if (self.calls and len(node.args) == 1 and not node.keywords
            and isinstance(node.args[0], (ast.Name, ast.BinOp))):
          # a name an argument holds, joined or not (#560): read where the
          # world is checked (`validate`), against the names a call passes
          sensor = self.name(node.args[0], "read")
          return _REFUSED if sensor is _REFUSED else ("readn", sensor, line)
        self.refuse(node, "read takes one sensor name in quotes")
        return _REFUSED
      if self.math and isinstance(fn, ast.Name) and fn.id in MATH:
        if node.keywords or len(node.args) != MATH[fn.id]:
          self.refuse(node, f"{fn.id} takes {MATH[fn.id]} number(s)")
          return _REFUSED
        return ("fn", fn.id, *[self.expr(a) for a in node.args])
      named = isinstance(fn, ast.Name) and (fn.id in self.verbs or self.callable(fn.id))
      self.refuse(node, "only read(...)" + (", sin, cos, atan2 and sqrt" if self.math else "")
                  + " may be used inside an expression; a verb is a statement of its own"
                  + (", and so is a call" if named and self.callable(fn.id) else "")
                  + (", or alone as an `if`'s test: `if verb(...):` or `if not verb(...):`"
                     if self.verb_tests and named else ""))
      return _REFUSED
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
      return ("bin", _BIN[type(node.op)], self.expr(node.left), self.expr(node.right))
    if isinstance(node, ast.BinOp):
      self.refuse(node, f"{_OP_TEXT.get(type(node.op), type(node.op).__name__)} is not "
                        "part of this language: numbers combine with + - * / alone")
      return _REFUSED
    if isinstance(node, ast.Attribute):
      self.refuse(node, _dotted_refusal(node))
      return _REFUSED
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
      return ("neg", self.expr(node.operand))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
      return ("not", self.expr(node.operand))
    if isinstance(node, ast.Compare):
      if len(node.ops) != 1 or type(node.ops[0]) not in _CMP:
        self.refuse(node, "compare two things with one of < <= > >= == !=")
        return _REFUSED
      return ("cmp", _CMP[type(node.ops[0])], self.expr(node.left),
              self.expr(node.comparators[0]))
    if isinstance(node, ast.BoolOp):
      # a chain of n nests n deep in what it builds, and in `_eval` running it
      if self.depth + len(node.values) - 1 > MAX_EXPR_DEPTH:
        self.refuse(node, f"an expression nests deeper than {MAX_EXPR_DEPTH}")
        return _REFUSED
      op = "and" if isinstance(node.op, ast.And) else "or"
      out = self.expr(node.values[0])
      for v in node.values[1:]:
        out = (op, out, self.expr(v))
      return out
    self.refuse(node, f"{type(node).__name__} is not part of this language")
    return _REFUSED

  def arg(self, node, name: str, kind: str) -> Any:
    """A verb argument: a quoted name, or a numeric expression. Anything
    else where the verb takes a name is refused here, once -- `validate`
    never refuses a `_REFUSED` again (#463: `fetch(None)` read both `a
    NoneType is not a number` and `tool takes a name in quotes`)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
      return ("str", node.value)
    if kind == "str":
      return self.name(node, name)
    return self.expr(node)

  # -- statements --
  def block(self, nodes, depth: int) -> tuple:
    return tuple(self.stmt(n, depth) for n in nodes)

  def stmt(self, node, depth: int) -> Any:
    if depth > MAX_NESTING:
      self.refuse(node, f"nested deeper than {MAX_NESTING}")
      return ("pass",)
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
      return self._verb(node, node.value)
    return self._stmt(node, depth)

  def cond(self, test) -> Any:
    """An `if`'s test: an expression -- or, where the world allows it
    (`WorldFacts.verb_tests`), a verb alone, `not` before it or not, run AS
    the test: its failing is read, never the end of the procedure."""
    neg, call = False, test
    if isinstance(call, ast.UnaryOp) and isinstance(call.op, ast.Not):
      neg, call = True, call.operand
    if (self.verb_tests and isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        and (call.func.id in self.verbs or self.callable(call.func.id))):
      verb = self._verb(test, call)
      return ("try", neg, verb) if verb[0] in ("verb", "call") else _REFUSED
    return self.expr(test)

  def _verb(self, node, call) -> tuple:
    """A verb call: ("verb", name, args, line) -- or a def's, ("call", name,
    args, line, "own" | "library"), its arguments read once every def's
    are known (`_link`) -- or ("pass",) refused."""
    fn = call.func
    if not isinstance(fn, ast.Name):
      self.refuse(node, "a statement is a verb call by its bare name")
      return ("pass",)
    if fn.id == "read":
      self.refuse(node, "read(...) is an expression; assign or compare it")
      return ("pass",)
    verb = self.verbs.get(fn.id)
    if verb is None and self.callable(fn.id):
      args: dict = {}
      line = getattr(node, "lineno", 0)
      where = "own" if fn.id in self.own else "library"
      self.sites.append((self.fn, fn.id, where, call, line, args))
      return ("call", fn.id, args, line, where)
    if verb is None:
      self.unknown = True
      self.refuse(node, f"unknown verb {fn.id!r} (have: {', '.join(self.verbs)})"
                  + ("" if not self.calls else
                     ", and no def of this source" + (" or procedure of your library"
                                                      if self.library is not None else "")
                     + " is named so"))
      return ("pass",)
    names = list(verb.args)
    args: dict = {}
    self.call = (fn.id, getattr(node, "lineno", 0))
    try:
      if len(call.args) > len(names):
        self.refuse(node, f"takes {len(names)} argument(s)")
      for name, value in zip(names, call.args):
        args[name] = self.arg(value, name, verb.args[name].kind)
      for kw in call.keywords:
        if kw.arg is None or kw.arg not in names:
          self.refuse(node, f"takes no {kw.arg or '**'}")
          continue
        if kw.arg in args:
          # by position and by keyword: the keyword overwrote the written
          # one, and a call the robot never wrote ran
          self.refuse(node, f"{kw.arg} is given twice")
          continue
        args[kw.arg] = self.arg(kw.value, kw.arg, verb.args[kw.arg].kind)
      missing = [n for n in names if n not in args and verb.args[n].default is None
                 and not verb.args[n].optional]
      if missing:
        self.refuse(node, f"needs {', '.join(missing)}")
    finally:
      self.call = None
    return ("verb", fn.id, args, getattr(node, "lineno", 0))

  def _stmt(self, node, depth: int) -> Any:
    if isinstance(node, ast.Assign):
      if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        self.refuse(node, "assign to one plain name")
        return ("pass",)
      name = node.targets[0].id
      why = name_fault(name, self.verbs, self.own if self.calls else ())
      if why:
        self.bad_name(node, name, why)
      self._sets(name, node)
      # ...its line, which names a fault in its arithmetic (#686)
      return ("set", name, self.expr(node.value), getattr(node, "lineno", 0))
    if isinstance(node, ast.AugAssign):
      if not isinstance(node.target, ast.Name) or type(node.op) not in _BIN:
        self.refuse(node, "augmented assignment is name op= expr with + - * /")
        return ("pass",)
      self._sets(node.target.id, node)
      if node.target.id in self.params:
        self.use(node.target.id, "number", node)
      return ("set", node.target.id,
              ("bin", _BIN[type(node.op)], ("var", node.target.id),
               self.expr(node.value)), getattr(node, "lineno", 0))
    if isinstance(node, ast.If):
      # ...and each arm's test's line, an `elif`'s its own
      arms = [(self.cond(node.test), self.block(node.body, depth + 1))]
      lines = [getattr(node, "lineno", 0)]
      orelse = node.orelse
      while len(orelse) == 1 and isinstance(orelse[0], ast.If):
        arms.append((self.cond(orelse[0].test),
                     self.block(orelse[0].body, depth + 1)))
        lines.append(getattr(orelse[0], "lineno", 0))
        orelse = orelse[0].orelse
      return ("if", tuple(arms), self.block(orelse, depth + 1), tuple(lines))
    if isinstance(node, ast.For):
      it = node.iter
      ok = (isinstance(node.target, ast.Name) and isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name) and it.func.id == "range"
            and len(it.args) == 1 and not it.keywords and not node.orelse
            and isinstance(it.args[0], ast.Constant)
            and isinstance(it.args[0].value, int)
            and not isinstance(it.args[0].value, bool))
      if not ok:
        self.refuse(node, "a loop is `for name in range(N)` with a literal N")
        return ("pass",)
      count = int(it.args[0].value)
      if not 0 <= count <= MAX_ITER:
        self.refuse(node, f"range({count}) is outside 0..{MAX_ITER}")
      self._sets(node.target.id, node)
      return ("repeat", node.target.id, count, self.block(node.body, depth + 1))
    if isinstance(node, ast.While):
      if node.orelse:
        self.refuse(node, "while takes no else")
      return ("while", self.expr(node.test), self.block(node.body, depth + 1),
              getattr(node, "lineno", 0))
    if isinstance(node, ast.Pass):
      return ("pass",)
    if isinstance(node, ast.Return):
      if node.value is not None:
        self.refuse(node, "return takes no value")
      return ("return",)
    self.refuse(node, f"{type(node).__name__} is not part of this language")
    return ("pass",)

  def _sets(self, name: str, node) -> None:
    if name in self.params:
      self.assigned.setdefault(self.fn, {}).setdefault(name, getattr(node, "lineno", 0))

  # -- the calls, once every def is read (#560) --
  def _link(self, first: str) -> dict:
    """Each def's calls, callees first: a call's arguments matched to its
    callee's and read as each holds a name or a number; each def's
    arguments typed by their uses; no def calling round to itself, none
    deeper than `MAX_CALL_DEPTH` below `first`. Returns each def's
    (name-holding arguments, verbs counted through its calls, depth)."""
    by_def: dict[str, list] = {}
    for site in self.sites:
      by_def.setdefault(site[0], []).append(site)
    order: list[str] = []
    state: dict[str, int] = {}
    looped = []

    def visit(d: str, path: list) -> None:
      state[d] = 1
      for site in by_def.get(d, ()):
        callee = site[1]
        if site[2] != "own":
          continue
        if state.get(callee) == 1:
          loop = path[path.index(callee):] + [callee] if callee in path else [d, callee]
          self.reasons.append(f"line {site[4]}: " + _round(loop))
          looped.append(loop)
          continue
        if callee not in state:
          visit(callee, path + [callee])
      state[d] = 2
      order.append(d)
    for d in self.own:
      if d not in state:
        visit(d, [d])
    if looped:
      return {}
    out: dict[str, tuple] = {}
    for d in order:
      params, defaults, _ = self.own[d]
      self.fn, self.params = d, params
      deepest, verbs = 0, 0
      for _, callee, where, call, line, args in by_def.get(d, ()):
        if where == "own":
          cparams, cdefaults, _ = self.own[callee]
          cnames, cnumbers, cverbs, cdepth = out[callee]
        else:
          linked = self.library[callee]
          if not isinstance(linked, Procedure):
            self.unknown = True                # its verbs are uncounted
            self.call = (callee, line)
            self.refuse(call, f"{callee} cannot run now -- "
                              + (linked[0] if linked else "it does not compile"))
            self.call = None
            continue
          cparams, cdefaults, cnames = linked.params, linked.defaults, linked.names
          cnumbers, cverbs, cdepth = linked.numbers, linked.verbs, linked.depth
        verbs += cverbs
        deepest = max(deepest, cdepth + 1)
        self.call = (callee, line)
        try:
          self._arguments(call, cparams, cdefaults, cnames, cnumbers, args)
        finally:
          self.call = None
      verbs += _count_verbs(self.bodies[d])
      uses = self.uses.get(d, {})
      names = frozenset(p for p in params if "name" in uses.get(p, {}))
      numbers = frozenset(p for p in params if "number" in uses.get(p, {})) - names
      for p in params:
        u = uses.get(p, {})
        if "name" in u and "number" in u:
          self.reasons.append(f"line {u['number']}: {p} holds a name (line {u['name']}), "
                              "and is used as a number")
        kind = "name" if p in names else "number" if "number" in u else None
        if p in defaults and kind is not None and (defaults[p][0] == "str") != (kind == "name"):
          self.reasons.append(f"line {self.own[d][2]}: {p} holds a {kind}, and its default is "
                              + ("a number" if kind == "name" else "a name"))
      for p, line in self.assigned.get(d, {}).items():
        if p in names:
          self.reasons.append(f"line {line}: {p} holds a name, which is never set")
      out[d] = (names, numbers, verbs, deepest)
    depth = out[first][3] if first in out else 0
    if depth > MAX_CALL_DEPTH:
      self.reasons.append(f"line {self.own[first][2]}: calls go {depth} deep below {first} "
                          f"({self._deepest(first, out, by_def)}); the most is {MAX_CALL_DEPTH}")
    return out

  def _arguments(self, call, params, defaults, names, numbers, args: dict) -> None:
    """One call's arguments, matched as a verb's are (by position, then by
    keyword) and read into `args`: a name where the callee's argument holds
    one, a number where it holds one, either where it uses it not at all."""
    kind = {p: "name" if p in names else "number" if p in numbers else None for p in params}
    if len(call.args) > len(params):
      self.refuse(call, f"takes {len(params)} argument(s)")
    for p, value in zip(params, call.args):
      args[p] = self._value(value, p, kind[p])
    for kw in call.keywords:
      if kw.arg is None or kw.arg not in params:
        self.refuse(call, f"takes no {kw.arg or '**'}")
        continue
      if kw.arg in args:
        self.refuse(call, f"{kw.arg} is given twice")
        continue
      args[kw.arg] = self._value(kw.value, kw.arg, kind[kw.arg])
    missing = [p for p in params if p not in args and p not in defaults]
    if missing:
      self.refuse(call, f"needs {', '.join(missing)}")

  def _value(self, node, param: str, kind: str | None) -> Any:
    if kind == "name":
      return self.name(node, param)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
      if kind == "number":
        self.refuse(node, f"{param} takes a number, not a name in quotes")
        return _REFUSED
      return ("str", node.value)               # unused by its def: a name or a number
    if kind is None and isinstance(node, ast.Name) and node.id in self.params:
      # ...and an argument of the caller's passed on as it is, typing it
      # neither way (read as a number, a name passed on was refused)
      return ("pass", node.id)
    return self.expr(node)

  def _deepest(self, d: str, out: dict, by_def: dict) -> str:
    chain = [d]
    while out.get(chain[-1], (0, 0, 0, 0))[3]:
      here = chain[-1]
      nxt = max((s for s in by_def.get(here, ()) if s[2] == "own" and s[1] in out),
                key=lambda s: out[s[1]][3], default=None)
      if nxt is None:
        break
      chain.append(nxt[1])
    return " > ".join(chain)


def _round(loop: list) -> str:
  """A def that calls round to itself, in words."""
  if len(loop) <= 2:
    return f"{loop[0]} calls itself: a def may not call itself, by way of another or not"
  return (f"{loop[0]} calls " + ", which calls ".join(loop[1:])
          + ": a def may not call itself, by way of another or not")


def parse(source: str, verbs: dict | None = None, verb_tests: bool = False,
          math: bool = False, calls: bool = False) -> Procedure:
  """Surface syntax -> Procedure, or Refused with every reason."""
  proc, reasons = _parse(source, verbs, verb_tests, math, calls)
  if reasons:
    raise Refused(reasons)
  return proc


def _parse(source: str, verbs: dict | None = None, verb_tests: bool = False,
           math: bool = False, calls: bool = False, library: dict | None = None,
           alone: bool = True) -> tuple:
  """The parse with its reasons, so `compile_procedure` can add the
  world's on top and refuse everything at once. `verbs`: what it may call
  (`steps.language`), `steps.VERBS` unsaid; `verb_tests`, whether a verb
  may be an `if`'s test; `math`, whether an expression may call `MATH`;
  `calls`, whether a source may hold several defs that call each other --
  and `library`'s procedures, by name (compiled, or the reasons one is
  not) -- and `alone`, whether the first def must run with nothing
  passing it arguments (all but a library's, #560)."""
  cap = []
  if len(source) > MAX_SOURCE_CHARS:
    cap = [f"the source is {len(source)} characters; the cap is {MAX_SOURCE_CHARS}"]
    if len(source) > MAX_PARSE_CHARS:
      raise Refused(cap)
  try:
    tree = ast.parse(source)
  except SyntaxError as e:
    # ...where one over the cap is no text that parses, the cap alone: the
    # served answer's source is cut one past it (`overseer`'s `define`)
    raise Refused(cap or [f"line {e.lineno or 0}: {e.msg}"]) from None
  except ValueError:
    # a lone surrogate (`\ud800` in the answer's JSON): UnicodeEncodeError
    # went out of the define and ended the served process (#500)
    raise Refused(cap or ["the source holds a character that is not text"]) from None
  except (MemoryError, RecursionError):
    # nesting past what Python's own parser holds ("parser stack overflowed")
    raise Refused(cap or ["the source nests too deeply to read"]) from None
  fns = tree.body
  if not calls and (len(fns) != 1 or not isinstance(fns[0], ast.FunctionDef)):
    raise Refused(cap + ["a procedure is exactly one `def name():`"])
  if calls and (not fns or not all(isinstance(fn, ast.FunctionDef) for fn in fns)):
    raise Refused(cap + ["a source is one def or more -- `def name(...):` -- and "
                         "nothing outside them"])
  p = _Parser(source, verbs, verb_tests, math, calls, library)
  p.reasons.extend(cap)
  p.bodies = {}
  first = fns[0].name
  for fn in fns:                           # every header first: a call may
    if fn.name in p.own:                   # name a def written below it
      p.refuse(fn, f"two defs are named {fn.name}")
      continue
    p.own[fn.name] = _header(p, fn, fn is fns[0])
  for fn in fns:
    params = p.own[fn.name][0]
    for x in params:
      if x in p.own:
        p.refuse(fn, f"{x!r} is a def's name, and an argument takes a name of its own")
  if alone:
    missing = [x for x in p.own[first][0] if x not in p.own[first][1]]
    if missing:
      # ...among several, most likely a helper written first, as a Python
      # script does (#640: 5 of 11 such first answers), which no default mends
      p.refuse(fns[0], f"{first} is the def that runs, and nothing passes it "
                       f"{', '.join(missing)}: "
                       + ("the first def runs, so put the one you mean to run first, or "
                          if len(fns) > 1 else "") + "give each a default")
  budgets, written = {}, {}
  for fn in fns:
    if fn.name in budgets:
      continue
    p.fn, p.params = fn.name, p.own[fn.name][0]
    budgets[fn.name], body = _budget(p, list(fn.body))
    if not body:
      p.refuse(fn, "the procedure has no steps" if fn.name == first
                   else f"{fn.name} has no steps")
    written[fn.name] = bool(body)
    p.bodies[fn.name] = p.block(body, 1)
  p.fn, p.params = "", ()
  out = p._link(first) if calls else {}
  names, numbers, verbs, depth = out.get(first, (frozenset(), frozenset(),
                                                 _count_verbs(p.bodies[first]), 0))
  if written[first] and not verbs and (out or not calls) and not (calls and p.unknown):
    # (a def that calls round to itself, or names a call it may not make, is
    # refused for that, and its verbs are not all counted)
    p.refuse(fns[0], "the procedure calls no verb")
  params, defaults, _ = p.own[first]
  defs = {d: Def(d, p.own[d][0], p.own[d][1], out.get(d, (frozenset(),))[0], p.bodies[d],
                 out.get(d, (frozenset(), frozenset()))[1])
          for d in p.bodies if d != first}
  linked = {s[1]: library[s[1]] for s in p.sites
            if s[2] == "library" and isinstance(library.get(s[1]), Procedure)}
  steps_budget, budget_s = budgets[first]
  return Procedure(name=first, source=source, body=p.bodies[first],
                   steps_budget=steps_budget, budget_s=budget_s,
                   verbs=verbs, roles={st.DEFAULT_ROLE: p.bodies[first]},
                   params=params, defaults=defaults, names=names, defs=defs,
                   linked=linked, depth=depth, numbers=numbers), p.reasons


def _header(p: _Parser, fn, first: bool) -> tuple:
  """A def's name and arguments: (its arguments, their defaults, its line).
  Where defs call none, a procedure takes none."""
  if not NAME_RE.match(fn.name):
    p.bad_name(fn, fn.name, name_rule())
  elif (p.calls and not first
        and (fn.name in RESERVED or fn.name in p.verbs or fn.name in MATH)):
    # a call by it would be the verb's: `wait(1)` in a source with a `def
    # wait` -- after the first only, which is run, never called by its own
    # source, so a kept procedure named before calls were is kept running
    p.refuse(fn, f"{fn.name!r} is the language's own name: a def takes a name of its own")
  a = fn.args
  line = getattr(fn, "lineno", 0)
  if not p.calls:
    if (a.args or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg
        or fn.decorator_list or fn.returns):
      p.refuse(fn, "a procedure takes no arguments and no decorators")
    return (), {}, line
  if (a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg or fn.decorator_list
      or fn.returns or any(x.annotation for x in a.args)):
    p.refuse(fn, "a def's arguments are plain names, with defaults or not: no *, **, / "
                 "or annotations, and no decorators")
  params = tuple(x.arg for x in a.args)
  for x in params:
    why = name_fault(x, p.verbs, math=True)
    if why:
      p.bad_name(fn, x, why)
  defaults = {}
  for x, d in zip(params[len(params) - len(a.defaults):], a.defaults):
    lit = _literal(d)
    if lit is None:
      p.refuse(fn, f"the default of {x} is a number or a name in quotes")
    else:
      defaults[x] = lit
  return params, defaults, line


def _literal(node):
  """A default: ("num", v) for a number, -1.5 too, or ("str", s) for a name
  in quotes; else None."""
  neg = isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
  value = node.operand if neg else node
  if isinstance(value, ast.Constant) and finite(value.value):
    return ("num", -float(value.value) if neg else float(value.value))
  if not neg and isinstance(value, ast.Constant) and isinstance(value.value, str):
    return ("str", value.value)
  return None


def _budget(p: _Parser, body: list) -> tuple:
  """A def's `budget(...)`, its first statement after any docstring: the
  budget and the body after it. The first def's is the run's, and a def it
  calls runs on that one (#560)."""
  if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
      and isinstance(body[0].value.value, str):
    body = body[1:]                        # a docstring is allowed and ignored
  steps_budget, budget_s = DEFAULT_STEPS, DEFAULT_BUDGET_S
  if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Call) \
      and isinstance(body[0].value.func, ast.Name) and body[0].value.func.id == "budget":
    call = body[0].value
    p.call = ("budget", getattr(call, "lineno", 0))
    if call.args:
      p.refuse(call, "takes keywords: budget(steps=N, seconds=S)")
    for kw in call.keywords:
      lit = kw.value.value if isinstance(kw.value, ast.Constant) else None
      if kw.arg == "steps" and isinstance(lit, int) and not isinstance(lit, bool):
        steps_budget = lit
      elif kw.arg == "seconds" and finite(lit):
        budget_s = float(lit)
      elif kw.arg == "seconds" and isinstance(lit, (int, float)) and not isinstance(lit, bool):
        p.refuse(call, "a number is too large to hold")       # as an argument's (#500)
      else:
        p.refuse(call, f"{kw.arg or '**'}=... is not a literal steps or seconds")
    if not 1 <= steps_budget <= MAX_STEPS:
      p.refuse(call, f"steps={steps_budget} is outside 1..{MAX_STEPS}")
    if not 0.0 < budget_s <= MAX_BUDGET_S:
      p.refuse(call, f"seconds={budget_s} is outside (0, {MAX_BUDGET_S:.0f}]")
    p.call = None
    body = body[1:]
  return (steps_budget, budget_s), body


def _count_verbs(block) -> int:
  """The verbs a block calls, written in it: a call's callee's are its
  caller's to add (`_Parser._link`)."""
  n = 0
  for s in block:
    if s[0] == "verb":
      n += 1
    elif s[0] == "if":
      n += (sum(_count_verbs(b) for _, b in s[1]) + _count_verbs(s[2])
            + sum(1 for c, _ in s[1] if c[0] == "try" and c[2][0] == "verb"))
    elif s[0] in ("repeat", "while"):
      n += _count_verbs(s[3 if s[0] == "repeat" else 2])
  return n


# ---- validation against the world -----------------------------------------------


def validate(proc: Procedure, facts: WorldFacts) -> list[str]:
  """What the parser could not know: names against THIS world. A literal
  argument is checked here exactly as a program's is (`steps.validate`); a
  computed one is checked when it is computed, as a failed step."""
  bad: list[str] = []

  def walk(block):
    for s in block:
      if s[0] == "verb":
        _, verb, args, line = s
        spec = st.language(facts)[verb]
        at = f"line {line}, {verb}: "          # the call, named once (#463)
        lit = {}
        for name, value in args.items():
          if value is _REFUSED or value[0] in NAMED:
            continue                           # the parser said why, once; a
          if value[0] == "str":                # name an argument holds, `_named`
            if spec.args[name].kind != "str":
              bad.append(f"{at}{name} takes a number, not a name in quotes")
            else:
              lit[name] = value[1]
          elif value[0] == "num":
            lit[name] = value[1]
          else:
            _sensors(value, bad, at)
        bad.extend(at + r for r in st.check_step(spec, lit, facts, partial=True, given=args))
      elif s[0] == "call":
        for value in s[2].values():
          if value is not _REFUSED and value[0] not in NAMED + ("str",):
            _sensors(value, bad, f"line {s[3]}, {s[1]}: ")
      elif s[0] == "if":
        for cond, body in s[1]:
          if cond[0] == "try":
            walk((cond[2],))
          else:
            _sensors(cond, bad)
          walk(body)
        walk(s[2])
      elif s[0] == "repeat":
        walk(s[3])
      elif s[0] == "while":
        _sensors(s[1], bad)
        walk(s[2])
      elif s[0] == "set":
        _sensors(s[2], bad)

  # The sensors are listed once a define: a refused define is one History
  # line of 400 characters, and a misspelt read in three calls listed them
  # three times and cut the reasons after it (#463)
  listed: list[bool] = []

  def _sensors(expr, out, at=""):
    if not isinstance(expr, tuple):
      return
    if expr[0] == "read" and expr[1] not in facts.sensors:
      have = "" if listed else f" (have: {', '.join(facts.sensors)})"
      listed.append(True)
      # outside a call (`x = ...`, `if`, `while`) the read's own line: with
      # none, a page's editor could mark no line (rooftop-media-2026 #441)
      out.append(f"{at or f'line {expr[2]}: '}unknown sensor {expr[1]!r}{have}")
    for sub in expr[1:]:
      if isinstance(sub, tuple):
        _sensors(sub, out, at)

  walk(proc.body)
  for d in proc.defs.values():
    walk(d.body)
  bad.extend(_named(proc, facts))
  return list(dict.fromkeys(bad))


#: A name an argument holds, and names joined (#560): read at the call.
NAMED = ("param", "join")


def _resolve(node, bound: dict) -> str | None:
  """A name node as the names a call passes make it, or None where an
  argument's name is not known before the run."""
  if node[0] == "str":
    return node[1]
  if node[0] == "param":
    return bound.get(node[1])
  if node[0] == "join":
    a, b = _resolve(node[1], bound), _resolve(node[2], bound)
    return None if a is None or b is None else a + b
  return None


def _named(proc: Procedure, facts: WorldFacts) -> list[str]:
  """Every name an argument holds, checked against the world where the
  calls that pass it say what it is (#560): from the def that runs, each
  call's names carried into its callee -- `put("middle", ...)` refused
  where `put` reaches with it -- each refusal naming the line that uses
  it and the calls it came through. A name no call can know is checked as
  the run makes it (`run_procedure_routine`)."""
  out: list[str] = []
  if facts is None:
    return out                               # no world to check a name against
  seen: set = set()
  table = st.language(facts)

  def where(owner, line, chain) -> tuple[str, str]:
    """The line that uses a name, in the library's procedure that holds it
    where that is another's, and the calls it came through."""
    at = f"line {line}" + (f" of {owner.name}" if owner is not proc else "")
    via = (" -- called at " + " > ".join(f"line {ln}, {n}" for ln, n in chain)) if chain else ""
    return at, via

  def exprs(node, owner, bound, chain):
    if not isinstance(node, tuple):
      return
    if node[0] == "readn":
      name = _resolve(node[1], bound)
      if name is not None and name not in facts.sensors:
        at, via = where(owner, node[2], chain)
        out.append(f"{at}: unknown sensor {name!r}{via}")
    for sub in node[1:]:
      exprs(sub, owner, bound, chain)

  def visit(owner: Procedure, body, bound: dict, chain: list) -> None:
    for s in body:
      if s[0] == "verb":
        _, verb, args, line = s
        spec = table.get(verb)
        if spec is None or (facts.verbs is not None and verb not in facts.verbs):
          continue                             # the verb's own refusal says it
        for name, value in args.items():
          if value is _REFUSED:
            continue
          if value[0] in NAMED:
            got = _resolve(value, bound)
            if got is None:
              continue
            if name == st.UNTIL:
              said = st.guard_sensor(got, facts)
            else:
              said = st.check_arg(spec, name, got, facts)
            at, via = where(owner, line, chain)
            out.extend(f"{at}, {verb}: {r}{via}" for r in said)
          else:
            exprs(value, owner, bound, chain)
      elif s[0] == "call":
        _, callee, args, line, kind = s
        target = owner.defs.get(callee) if kind == "own" else owner.linked.get(callee)
        if target is None:
          continue
        home = owner if kind == "own" else target
        passed = {}
        for p in target.names:
          value = args.get(p, target.defaults.get(p))
          passed[p] = None if value is None or value is _REFUSED else _resolve(value, bound)
        key = (id(target), tuple(sorted(passed.items())))
        if key not in seen:
          seen.add(key)
          visit(home, target.body, passed, chain + [(line, callee)])
        for value in args.values():
          if value is not _REFUSED and value[0] not in NAMED + ("str",):
            exprs(value, owner, bound, chain)
      elif s[0] == "if":
        for cond, arm in s[1]:
          if cond[0] == "try":
            visit(owner, (cond[2],), bound, chain)
          else:
            exprs(cond, owner, bound, chain)
          visit(owner, arm, bound, chain)
        visit(owner, s[2], bound, chain)
      elif s[0] == "repeat":
        visit(owner, s[3], bound, chain)
      elif s[0] == "while":
        exprs(s[1], owner, bound, chain)
        visit(owner, s[2], bound, chain)
      elif s[0] == "set":
        exprs(s[2], owner, bound, chain)

  visit(proc, proc.body, {p: _resolve(proc.defaults[p], {}) for p in proc.names
                          if p in proc.defaults}, [])
  return out


def compile_procedure(source: str, facts: WorldFacts, library: dict | None = None,
                      alone: bool = True) -> Procedure:
  """Parse and validate, and refuse with EVERY reason from both. Where the
  world allows calls, `library` is what a call may name beside the
  source's own defs (`Library`), and `alone` whether its first def must run
  with nothing passing it arguments (#560)."""
  proc, reasons = _parse(source, st.language(facts), bool(getattr(facts, "verb_tests", False)),
                         bool(getattr(facts, "math", False)),
                         bool(getattr(facts, "calls", False)), library, alone)
  reasons = reasons + validate(proc, facts)
  if reasons:
    raise Refused(list(dict.fromkeys(reasons)))
  return proc


# ---- the interpreter: a routine ------------------------------------------------


class _Stop(Exception):
  def __init__(self, why: str, at: tuple | None = None) -> None:
    super().__init__(why)
    self.why = why
    #: where it was, for a fault the run names as a step: a call's argument
    #: it could not compute -- the call's line, its callee and where the
    #: call ran (`At`, #560) -- or a statement's own arithmetic, `expr`
    #: (#686: it read "line ?")
    self.at = at


class _Return(Exception):
  pass


class _Failed(Exception):
  """A verb that failed inside a call an `if` tests (#560): it ends that
  call, which its test reads as false."""


#: What ends a whole run, a call an `if` tests too: a failed step where no
#: test reads it -- and a fall, which every test reads as one (#671) -- the
#: run's own limits, and its robot's death (#671).
LIMITS = ("failed", "interrupted", "budget", "steps", "statements", "died")


def _truth(v: float) -> bool:
  return bool(v)


def _name(node, env: dict) -> str:
  """A name node's name as the run has it (#560)."""
  if node[0] == "str":
    return node[1]
  if node[0] == "param":
    if not isinstance(env.get(node[1]), str):
      raise _Stop(f"{node[1]!r} holds no name")
    return env[node[1]]
  if node[0] == "join":
    return _name(node[1], env) + _name(node[2], env)
  raise _Stop("a name is needed here")


def _eval(expr, env: dict, life, facts: WorldFacts) -> float:
  kind = expr[0]
  if kind == "num":
    return float(expr[1])
  if kind == "var":
    if expr[1] not in env:
      raise _Stop(f"{expr[1]!r} was read before it was set")
    if isinstance(env[expr[1]], str):
      raise _Stop(f"{expr[1]!r} holds a name, not a number")
    return float(env[expr[1]])
  if kind in ("read", "readn"):
    from pluggybot.procedure import axes
    name = expr[1] if kind == "read" else _name(expr[1], env)
    sensor = axes.SENSORS.get(name)
    if sensor is None or (kind == "readn" and name not in facts.sensors):
      raise _Stop(f"unknown sensor {name!r}")
    try:
      return float(sensor.read(life))
    except axes.NoReading as no:
      # a verb's report read before the verb ran (issue #493), or a number
      # it could not give (#630): said, never a number made up
      raise _Stop(str(no)) from None
  if kind == "bin":
    a, b = _eval(expr[2], env, life, facts), _eval(expr[3], env, life, facts)
    op = expr[1]
    if op == "+":
      return a + b
    if op == "-":
      return a - b
    if op == "*":
      return a * b
    if b == 0.0:
      raise _Stop("division by zero")
    return a / b
  if kind == "fn":
    args = [_eval(a, env, life, facts) for a in expr[2:]]
    if expr[1] == "sqrt" and args[0] < 0.0:
      raise _Stop(f"sqrt of a negative number ({args[0]:g})")
    return float({"sin": math.sin, "cos": math.cos, "atan2": math.atan2,
                  "sqrt": math.sqrt}[expr[1]](*args))
  if kind == "neg":
    return -_eval(expr[1], env, life, facts)
  if kind == "not":
    return 0.0 if _truth(_eval(expr[1], env, life, facts)) else 1.0
  if kind == "cmp":
    a, b = _eval(expr[2], env, life, facts), _eval(expr[3], env, life, facts)
    op = expr[1]
    ok = {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b,
          "==": a == b, "!=": a != b}[op]
    return 1.0 if ok else 0.0
  if kind == "and":
    return (_eval(expr[2], env, life, facts)
            if _truth(_eval(expr[1], env, life, facts)) else 0.0)
  if kind == "or":
    a = _eval(expr[1], env, life, facts)
    return a if _truth(a) else _eval(expr[2], env, life, facts)
  raise _Stop(f"cannot evaluate {kind}")


def run_procedure_routine(life, proc: Procedure, facts: WorldFacts,
                          into: dict | None = None) -> Routine:
  """Interpret a validated procedure as a routine. The result has the shape a
  program's run has (steps.run_program_routine), so the errand around it and
  the evaluator read both alike: one verdict per verb executed, honest about
  how far it got and why it stopped. `into` is filled as the run goes, so a
  caller that closes the routine still has its record (#530: a try its time
  limit cut), the verb in flight under `inFlight`.

  A CALL (#560) runs its callee on the run's one budget, its verbs counted
  with every other, a step that ran inside one saying where (`in`, and
  `entry` for the library's procedure whose source holds its line); a
  verb failing in it ends the whole run as one failing anywhere does --
  unless the call is an `if`'s test, which then reads it false, as it reads
  a fault in its arithmetic, and the run goes on. A fall ends the run
  wherever it comes, and a death too (`stopped: died`, #671).

  `total` is the calls MADE, `completed` the calls the run went on past --
  every call made but one that ended it, a test that read false among them
  -- and `testsFalse` how many tests read false (#629)."""
  reasons = validate(proc, facts)
  if not proc.alone:
    reasons = reasons + [f"{proc.name} {called_only(proc)}"]
  if reasons:
    raise Refused(reasons)
  result: dict[str, Any] = into if into is not None else {}
  result.update({"program": proc.name, "role": st.DEFAULT_ROLE, "total": 0,
                 "completed": 0, "testsFalse": 0, "steps": [], "ok": False,
                 "source": proc.source})
  env: dict = {p: d[1] for p, d in proc.defaults.items()}
  t0 = float(life.data.time)
  count = 0
  ran = 0
  #: each callee's variables, as it last ended (`callee.name`)
  inner: dict[str, float] = {}

  @dataclass(frozen=True)
  class At:
    """Where a statement runs: the procedure whose source holds it, its
    def, how many calls deep, and whether a call an `if` tests holds it."""
    owner: Procedure
    fn: str
    depth: int = 0
    tested: bool = False

    def said(self) -> dict:
      return {**({"in": self.fn} if self.depth else {}),
              **({"entry": self.owner.name} if self.owner is not proc else {})}

    def text(self) -> str:
      return ((f", in {self.fn}" if self.depth else "")
              + (f" of {self.owner.name}" if self.owner is not proc else ""))

  def verb_step(verb: str, args: dict, line: int, env: dict, at: At,
                tested: bool = False) -> Routine:
    """One verb run: True when it did what it said. A failure ends the
    procedure -- unless the verb is an `if`'s test (`tested`), when it is
    recorded and read, or a call held by one (`at.tested`) ends there.
    ⚠ A FALL ENDS IT WHATEVER TESTS IT (#671; `st.run_verb`), as a fall
    fails a job's try, and a dead robot begins no verb: dead as the last one
    ended, or while its interrupt was asked, which steps the world -- and a
    dead robot is never asked one."""
    nonlocal count
    if (count and not st.dead(life) and (yield from life.interrupted_routine())
        and not st.dead(life)):
      raise _Stop("interrupted")
    if st.dead(life):
      raise _Stop("died")
    if float(life.data.time) - t0 > proc.budget_s:
      raise _Stop("budget")
    if count >= proc.steps_budget:
      raise _Stop("steps")
    spec = st.language(facts)[verb]
    values: dict = {}
    try:
      for name, value in args.items():
        values[name] = (_name(value, env) if value[0] in NAMED + ("str",)
                        else _eval(value, env, life, facts))
      # a computed argument is checked here, against the same rules a literal
      # met at compile time
      bad = st.check_step(spec, values, facts)
    except _Stop as fault:
      # ...and one it could not compute fails this call's step, named by its
      # line and verb as every refusal inside a call is (#463), where it was
      # "it stopped at an expression", no line, no call
      bad = [fault.why]
    count += 1
    result["total"] = count
    result["inFlight"] = {"line": line, "verb": verb, **at.said()}
    # a watcher's word that a verb began (#530: the line a person's page
    # lights); a life with none says nothing more
    began = getattr(life, "verb_began", None)
    if began is not None:
      began(count, line, verb, values)
    life._say(f"PROCEDURE {proc.name} step {count} (line {line}{at.text()}): "
              f"{st.Step(verb, values).describe()}")
    if bad:
      st.forget_readings(life, verb)
      verdict = {"ok": False, "reason": "; ".join(bad)}
    else:
      verdict = yield from st.run_verb(life, spec, values,
                                       {"procedure": proc.name, "n": count,
                                        "line": line, **at.said()},
                                       until=t0 + proc.budget_s)
    read = not verdict.get("ok") and at.tested
    tests = (tested or read) and not verdict.get("fell")
    entry = {"i": count - 1, "verb": verb, "line": line, **at.said(), **verdict,
             **({"tested": True} if tests else {})}
    result["steps"].append(entry)
    del result["inFlight"]
    if verdict.get("stopped") in ("interrupted", "died"):
      # the robot's own interrupt ended a walk inside it (issue #381,
      # `run_verb`): an abort, as between two verbs, never a failed step --
      # and so is the robot's death inside it (#671)
      raise _Stop(verdict["stopped"])
    if not verdict.get("ok") and tests:
      # ...a call the run goes on past: set on a success alone, `completed`
      # was the NUMBER of the last call that did what it said, and a run
      # whose last calls were tests that read false was graded cut short
      result["completed"] = count
      result["testsFalse"] += 1
      life._say(f"PROCEDURE {proc.name} step {count} (line {line}{at.text()}, {verb}) did "
                "not do what it said, as its test read"
                + (f" -- {verdict['reason']}" if verdict.get("reason") else ""))
      if not tested:
        raise _Failed()
      return False
    if not verdict.get("ok"):
      result["failedAt"] = count - 1
      life._say(f"PROCEDURE {proc.name} failed at step {count} (line {line}{at.text()}, {verb})"
                + (f" -- {verdict['reason']}" if verdict.get("reason") else ""),
                detail=verdict.get("trace", ""))
      raise _Stop("failed")
    result["completed"] = count
    return True

  def fault(why: str, line: int, verb: str, at: At) -> None:
    """A fault a test reads -- in the arithmetic a call it tests ran, or in
    computing that call's arguments: a step of its own that failed, and the
    run goes on (no count: no verb ran)."""
    result["steps"].append({"verb": verb, "line": line, **at.said(), "ok": False,
                            "reason": why, "tested": True})
    result["testsFalse"] += 1
    life._say(f"PROCEDURE {proc.name} line {line}{at.text()}, {verb} did not do what it "
              f"said, as its test read -- {why}")

  def call(node, env: dict, at: At, tested: bool = False) -> Routine:
    """A call (#560): True when its callee ran to its end or a `return`."""
    _, name, args, line, kind = node
    target = at.owner.defs[name] if kind == "own" else at.owner.linked[name]
    home = at.owner if kind == "own" else target
    held = at.tested or tested
    sub: dict = {}
    try:
      for p in target.params:
        value = args.get(p, target.defaults.get(p))
        if value[0] == "pass":
          if value[1] not in env:
            raise _Stop(f"{value[1]!r} was read before it was set")
          sub[p] = env[value[1]]
        else:
          sub[p] = (_name(value, env) if value[0] in NAMED + ("str",)
                    else _eval(value, env, life, facts))
    except _Stop as no:
      # an argument it could not compute fails the call, at its own line
      if not held:
        raise _Stop(no.why, (line, name, at)) from None
      fault(no.why, line, name, at)
      if not tested:
        raise _Failed() from None
      return False
    inside = At(home, name, at.depth + 1, held)
    bound = dict(sub)
    try:
      yield from block(target.body, sub, inside)
    except _Return:
      pass
    except _Failed:
      if not tested:
        raise
      return False
    except _Stop as stop:
      if not tested or stop.why in LIMITS:
        raise
      # a fault in the arithmetic of a call a test reads (an argument's is
      # read where it is, `fault`): the call is false, and the run goes on
      # -- named by the line it came at, in the def that ran it (#686: by
      # the call's, beside the def's name)
      fault(stop.why, *(stop.at or (line, "expr", inside)))
      return False
    finally:
      # a def's own variables, and an argument it set; keyed by the def --
      # in another of the library's procedures, by that procedure too, as
      # two of them may each keep a def of one name
      key = (name if home is proc else home.name if name == home.name
             else f"{home.name}.{name}")
      for k, v in sub.items():
        if isinstance(v, float) and (k not in target.params or v != bound[k]):
          inner[f"{key}.{k}"] = round(v, 6)
    return True

  def value(expr, line: int, env: dict, at: At) -> float:
    """What a statement's own expression computes: a fault in it named by
    the statement's line, in the def that ran it (#686: "line ?", where
    only a call's arguments carried theirs)."""
    try:
      return _eval(expr, env, life, facts)
    except _Stop as no:
      raise _Stop(no.why, (line, "expr", at)) from None

  def block(stmts, env: dict, at: At) -> Routine:
    nonlocal ran
    for s in stmts:
      ran += 1
      if ran > MAX_STATEMENTS:
        raise _Stop("statements")
      kind = s[0]
      if kind == "verb":
        yield from verb_step(s[1], s[2], s[3], env, at)
      elif kind == "call":
        yield from call(s, env, at)
      elif kind == "set":
        env[s[1]] = value(s[2], s[3], env, at)
      elif kind == "if":
        for (cond, body), line in zip(s[1], s[3]):
          if cond[0] == "try" and cond[2][0] == "call":
            held = (yield from call(cond[2], env, at, tested=True)) != cond[1]
          elif cond[0] == "try":
            _, neg, (_, verb, args, line) = cond
            held = (yield from verb_step(verb, args, line, env, at, tested=True)) != neg
          else:
            held = _truth(value(cond, line, env, at))
          if held:
            yield from block(body, env, at)
            break
        else:
          yield from block(s[2], env, at)
      elif kind == "repeat":
        for i in range(s[2]):
          env[s[1]] = float(i)
          yield from block(s[3], env, at)
      elif kind == "while":
        n = 0
        while _truth(value(s[1], s[3], env, at)):
          n += 1
          if n > MAX_ITER:
            raise _Stop("loop-cap", (s[3], "expr", at))
          yield from block(s[2], env, at)
      elif kind == "return":
        raise _Return()
      # "pass": nothing

  try:
    try:
      yield from block(proc.body, env, At(proc, proc.name))
      result["ok"] = True
    except _Return:
      result["ok"] = True
    except _Stop as stop:
      if stop.why != "failed":
        result["stopped"] = stop.why
        if stop.why not in LIMITS:
          # an evaluation fault is a failed step of its own, so the record
          # says where the arithmetic went wrong -- in a call's arguments,
          # at the call's line (#560), in a statement's own, at its (#686)
          result["total"] = count + 1
          result["failedAt"] = count
          line, verb, at = stop.at if stop.at else (None, "expr", None)
          result["steps"].append({"i": count, "verb": verb,
                                  **({"line": line, **at.said()} if line else {}),
                                  "ok": False, "reason": stop.why})
          life._say(f"PROCEDURE {proc.name} stopped: {stop.why}")
  finally:
    # ...whatever ended it, a close from outside included (no yield here)
    result["seconds"] = round(float(life.data.time) - t0, 2)
    # The locals as they stood when the run ended: a procedure's variables
    # are its readout (issue #227) -- what it read and computed reaches the
    # mind through these, and through nothing else; a callee's, as it last
    # ended, beside them (#560)
    result["locals"] = {**{k: round(float(v), 6) for k, v in env.items()
                           if isinstance(v, float)
                           and (k not in proc.params or v != proc.defaults[k][1])}, **inner}
    # ...and a watch it began ends with it (#554): no picture for nobody;
    # and what its failed calls said (#686)
    st.end_run(life, watching="watch" in (facts.verbs or ()))
  return result
