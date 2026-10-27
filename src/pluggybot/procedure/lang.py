"""The procedure language: conditionals, bounded loops and arithmetic over
the step vocabulary (issue #166, rung two of agent-written code).

Python-SHAPED, never Python-RUN. The surface syntax is a `def` with a body,
because a model writes that far better than a bespoke dialect and a garbled
answer then becomes a measurement instead of a tax -- but the text is
PARSED (with `ast`) into this module's own small tree and interpreted by a
routine; nothing here calls `exec`, `eval` or `compile`, and a program that
uses any construct outside the grammar is refused with the line it broke,
before a single step runs. The grammar:

    def go_and_look():                      # one def, no arguments
        budget(steps=60, seconds=300)       # optional, first; capped by code
        n = 0                               # locals, arithmetic
        fetch("module_lcd")                 # a verb, arguments literal or computed
        for i in range(3):                  # a literal count, <= MAX_ITER
            drive(0.0, 0.8, 1.5)
            if read("bumper") < 1:
                wait(2)
        while read("shoulder") < 1.0 and n < 10:      # capped at MAX_ITER
            move("shoulder", read("shoulder") + 0.1)  # the motor level: a ramped setpoint
            n = n + 1
        stow()

  statements   verb(...)  name = expr  name += expr  if/elif/else  for name in
               range(N)  while cond  pass  return  budget(...)
  expressions  numbers, True/False, locals, read("sensor"), + - * / unary -,
               comparisons, and/or/not, parentheses
  nothing else no strings outside a verb's or read's argument, no attribute
               access, no calls but verbs and read, no imports, no I/O

Totality by construction: every loop is bounded (a literal count, or the
cap), a program declares a step budget and a sim-time budget and the
vocabulary caps both, and the interpreter checks them at every verb -- which
is a safe point (steps.py). Abort means stow, as for every program.

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
MAX_SOURCE_CHARS = 1500
MAX_NESTING = 4
#: How deep an expression nests. Each level was one frame of the parser's
#: recursion, and ~980 unary minuses (a source of 1,000 characters) raised
#: RecursionError out of the define and ended the served process (#500).
MAX_EXPR_DEPTH = 50
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
#: The verbs that need the claw (issue #407): what `references` says a
#: procedure reaches for -- a `pick` fetches it onto an empty fork (#353).
CLAW_TOOL_VERBS = frozenset(st.CLAW_VERBS)
#: Names a local may not shadow: the language's own words and every verb.
RESERVED = frozenset({"read", "budget", "range", "def", "and", "or", "not",
                      "if", "elif", "else", "for", "while", "in", "return",
                      "pass", "True", "False", "None"})

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
class Procedure:
  """A compiled procedure: its source (what the robot wrote, what rides the
  wire and the library) and the tree the interpreter walks."""

  name: str
  source: str
  body: tuple = ()
  steps_budget: int = DEFAULT_STEPS
  budget_s: float = DEFAULT_BUDGET_S
  #: how many verb calls the body contains statically -- a rough size
  verbs: int = 0
  roles: dict = field(default_factory=dict)

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

  def references(self) -> dict:
    """What this procedure reaches for: the axes a `move` names, the sensors
    a `read` names, and the modules a `fetch` names -- in source order,
    deduplicated (issue #324).

    ⚠ `tools` IS NOT REDUNDANT WITH `axes`. The high-level verbs declare no
    tool: `draw` needs the pen, `pick`/`place`/`grip` need the claw, and
    `Verb` carries nothing that says so -- each checks inside its own body.
    So a procedure of `fetch("module_pen"); draw(...); stow()` names no axis
    at all, and reading its needs off axes alone answered "none", which is
    worse than saying nothing. A `fetch` target is the procedure declaring
    which tool it is about, and it is right there in the source -- and so
    is a `pick`, which fetches the claw onto an empty fork (issue #353).

    PURE: the tree only, no world. Which MODULE a sensor needs is
    `axes.SENSORS[...].requires`, and resolving that is the caller's job
    (`Library.as_context`).

    Every ROLE is walked, not just `body`: they are the same tuple for
    anything `define` produces (single-role), and this cannot silently
    under-report if that ever stops being true.
    """
    axes_named: list[str] = []
    sensors: list[str] = []
    tools: list[str] = []

    def expr(e):
      if not isinstance(e, tuple):
        return
      if e[0] == "read" and isinstance(e[1], str):
        sensors.append(e[1])
      for sub in e[1:]:
        expr(sub)

    def walk(block):
      for st_ in block:
        if st_[0] == "verb":
          for name, value in st_[2].items():
            if value[0] == "str":
              if st_[1] == "move" and name == "axis":
                axes_named.append(value[1])
              elif st_[1] == "fetch" and name == "tool":
                tools.append(value[1])
            expr(value)
          if st_[1] in CLAW_TOOL_VERBS:
            tools.append("module_claw")
        elif st_[0] == "set":
          expr(st_[2])
        elif st_[0] == "if":
          for cond, body in st_[1]:
            if cond[0] == "try":
              walk((cond[2],))
            else:
              expr(cond)
            walk(body)
          walk(st_[2])
        elif st_[0] == "repeat":
          walk(st_[3])
        elif st_[0] == "while":
          expr(st_[1])
          walk(st_[2])
    for block in (self.roles.values() or (self.body,)):
      walk(block)
    return {"axes": tuple(dict.fromkeys(axes_named)),
            "sensors": tuple(dict.fromkeys(sensors)),
            "tools": tuple(dict.fromkeys(tools))}

  def as_dict(self) -> dict:
    return {"name": self.name, "source": self.source,
            "budgetS": self.budget_s, "stepsBudget": self.steps_budget}

  @classmethod
  def from_dict(cls, spec: dict, facts: WorldFacts | None = None) -> "Procedure":
    return parse(str(spec.get("source", "")))


# ---- parsing: surface syntax -> the tree, or Refused ----------------------------


class _Parser:
  def __init__(self, source: str, verbs: dict | None = None,
               verb_tests: bool = False, math: bool = False) -> None:
    self.source = source
    #: whether a verb may stand alone as an `if`'s test (`WorldFacts.verb_tests`)
    self.verb_tests = verb_tests
    #: whether an expression may call `MATH`'s functions (`WorldFacts.math`)
    self.math = math
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
      if not NAME_RE.match(node.id) or node.id in RESERVED or node.id in self.verbs:
        self.refuse(node, f"{node.id!r} is not a name this language allows")
      return ("var", node.id)
    if isinstance(node, ast.Call):
      fn = node.func
      if isinstance(fn, ast.Name) and fn.id == "read":
        if (len(node.args) != 1 or node.keywords
            or not isinstance(node.args[0], ast.Constant)
            or not isinstance(node.args[0].value, str)):
          self.refuse(node, "read takes one sensor name in quotes")
          return _REFUSED
        # its line: a read outside a call is refused on it (`validate`)
        return ("read", node.args[0].value, getattr(node, "lineno", 0))
      if self.math and isinstance(fn, ast.Name) and fn.id in MATH:
        if node.keywords or len(node.args) != MATH[fn.id]:
          self.refuse(node, f"{fn.id} takes {MATH[fn.id]} number(s)")
          return _REFUSED
        return ("fn", fn.id, *[self.expr(a) for a in node.args])
      self.refuse(node, "only read(...)" + (", sin, cos, atan2 and sqrt" if self.math else "")
                  + " may be used inside an expression; a verb is a statement of its own"
                  + (", or alone as an `if`'s test: `if verb(...):` or `if not verb(...):`"
                     if self.verb_tests and isinstance(fn, ast.Name) and fn.id in self.verbs
                     else ""))
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
      self.refuse(node, f"{name} takes a name in quotes")
      return _REFUSED
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
        and call.func.id in self.verbs):
      verb = self._verb(test, call)
      return ("try", neg, verb) if verb[0] == "verb" else _REFUSED
    return self.expr(test)

  def _verb(self, node, call) -> tuple:
    """A verb call: ("verb", name, args, line), or ("pass",) refused."""
    fn = call.func
    if not isinstance(fn, ast.Name):
      self.refuse(node, "a statement is a verb call by its bare name")
      return ("pass",)
    if fn.id == "read":
      self.refuse(node, "read(...) is an expression; assign or compare it")
      return ("pass",)
    verb = self.verbs.get(fn.id)
    if verb is None:
      self.refuse(node, f"unknown verb {fn.id!r} (have: {', '.join(self.verbs)})")
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
      if not NAME_RE.match(name) or name in RESERVED or name in self.verbs:
        self.refuse(node, f"{name!r} is not a name this language allows "
                          "(it is reserved)")
      return ("set", name, self.expr(node.value))
    if isinstance(node, ast.AugAssign):
      if not isinstance(node.target, ast.Name) or type(node.op) not in _BIN:
        self.refuse(node, "augmented assignment is name op= expr with + - * /")
        return ("pass",)
      return ("set", node.target.id,
              ("bin", _BIN[type(node.op)], ("var", node.target.id),
               self.expr(node.value)))
    if isinstance(node, ast.If):
      arms = [(self.cond(node.test), self.block(node.body, depth + 1))]
      orelse = node.orelse
      while len(orelse) == 1 and isinstance(orelse[0], ast.If):
        arms.append((self.cond(orelse[0].test),
                     self.block(orelse[0].body, depth + 1)))
        orelse = orelse[0].orelse
      return ("if", tuple(arms), self.block(orelse, depth + 1))
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


def parse(source: str, verbs: dict | None = None, verb_tests: bool = False,
          math: bool = False) -> Procedure:
  """Surface syntax -> Procedure, or Refused with every reason."""
  proc, reasons = _parse(source, verbs, verb_tests, math)
  if reasons:
    raise Refused(reasons)
  return proc


def _parse(source: str, verbs: dict | None = None, verb_tests: bool = False,
           math: bool = False) -> tuple:
  """The parse with its reasons, so `compile_procedure` can add the
  world's on top and refuse everything at once. `verbs`: what it may call
  (`steps.language`), `steps.VERBS` unsaid; `verb_tests`, whether a verb
  may be an `if`'s test; `math`, whether an expression may call `MATH`."""
  if len(source) > MAX_SOURCE_CHARS:
    raise Refused([f"the source is {len(source)} characters; the cap is "
                   f"{MAX_SOURCE_CHARS}"])
  try:
    tree = ast.parse(source)
  except SyntaxError as e:
    raise Refused([f"line {e.lineno or 0}: {e.msg}"]) from None
  except ValueError:
    # a lone surrogate (`\ud800` in the answer's JSON): UnicodeEncodeError
    # went out of the define and ended the served process (#500)
    raise Refused(["the source holds a character that is not text"]) from None
  except (MemoryError, RecursionError):
    # nesting past what Python's own parser holds ("parser stack overflowed")
    raise Refused(["the source nests too deeply to read"]) from None
  if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
    raise Refused(["a procedure is exactly one `def name():`"])
  fn = tree.body[0]
  p = _Parser(source, verbs, verb_tests, math)
  if not NAME_RE.match(fn.name):
    p.refuse(fn, f"{fn.name!r} is not a name this language allows "
                 "(lowercase, digits, underscores, 32 characters)")
  a = fn.args
  if (a.args or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg
      or fn.decorator_list or fn.returns):
    p.refuse(fn, "a procedure takes no arguments and no decorators")
  body = list(fn.body)
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
  if not body:
    p.refuse(fn, "the procedure has no steps")
  compiled = p.block(body, 1)
  verbs = _count_verbs(compiled)
  if body and not verbs:
    p.refuse(fn, "the procedure calls no verb")
  return Procedure(name=fn.name, source=source, body=compiled,
                   steps_budget=steps_budget, budget_s=budget_s,
                   verbs=verbs, roles={st.DEFAULT_ROLE: compiled}), p.reasons


def _count_verbs(block) -> int:
  n = 0
  for s in block:
    if s[0] == "verb":
      n += 1
    elif s[0] == "if":
      n += (sum(_count_verbs(b) for _, b in s[1]) + _count_verbs(s[2])
            + sum(1 for c, _ in s[1] if c[0] == "try"))
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
          if value is _REFUSED:
            continue                           # the parser said why, once
          if value[0] == "str":
            if spec.args[name].kind != "str":
              bad.append(f"{at}{name} takes a number, not a name in quotes")
            else:
              lit[name] = value[1]
          elif value[0] == "num":
            lit[name] = value[1]
          else:
            _sensors(value, bad, at)
        bad.extend(at + r for r in st.check_step(spec, lit, facts, partial=True, given=args))
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
  return list(dict.fromkeys(bad))


def compile_procedure(source: str, facts: WorldFacts) -> Procedure:
  """Parse and validate, and refuse with EVERY reason from both."""
  proc, reasons = _parse(source, st.language(facts), bool(getattr(facts, "verb_tests", False)),
                         bool(getattr(facts, "math", False)))
  reasons = reasons + validate(proc, facts)
  if reasons:
    raise Refused(list(dict.fromkeys(reasons)))
  return proc


# ---- the interpreter: a routine ------------------------------------------------


class _Stop(Exception):
  def __init__(self, why: str) -> None:
    super().__init__(why)
    self.why = why


class _Return(Exception):
  pass


def _truth(v: float) -> bool:
  return bool(v)


def _eval(expr, env: dict, life, facts: WorldFacts) -> float:
  kind = expr[0]
  if kind == "num":
    return float(expr[1])
  if kind == "var":
    if expr[1] not in env:
      raise _Stop(f"{expr[1]!r} was read before it was set")
    return float(env[expr[1]])
  if kind == "read":
    from pluggybot.procedure import axes
    sensor = axes.SENSORS.get(expr[1])
    if sensor is None:
      raise _Stop(f"unknown sensor {expr[1]!r}")
    if sensor.requires and st._carried(life) != sensor.requires:
      raise _Stop(f"sensor {expr[1]!r} needs {sensor.requires} on the fork")
    try:
      return float(sensor.read(life))
    except axes.NoReading as no:
      # a verb's report read before the verb ran (issue #493): said, never
      # a number made up
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
  limit cut), the verb in flight under `inFlight`."""
  reasons = validate(proc, facts)
  if reasons:
    raise Refused(reasons)
  result: dict[str, Any] = into if into is not None else {}
  result.update({"program": proc.name, "role": st.DEFAULT_ROLE, "total": 0,
                 "completed": 0, "steps": [], "ok": False, "source": proc.source})
  env: dict[str, float] = {}
  t0 = float(life.data.time)
  count = 0

  def verb_step(verb: str, args: dict, line: int, tested: bool = False) -> Routine:
    """One verb run: True when it did what it said. A failure ends the
    procedure -- unless the verb is an `if`'s test (`tested`), when it is
    recorded and read."""
    nonlocal count
    if count and life.interrupted():
      raise _Stop("interrupted")
    if float(life.data.time) - t0 > proc.budget_s:
      raise _Stop("budget")
    if count >= proc.steps_budget:
      raise _Stop("steps")
    spec = st.language(facts)[verb]
    values: dict = {}
    try:
      for name, value in args.items():
        values[name] = value[1] if value[0] == "str" else _eval(value, env, life, facts)
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
    result["inFlight"] = {"line": line, "verb": verb}
    # a watcher's word that a verb began (#530: the line a person's page
    # lights); a life with none says nothing more
    began = getattr(life, "verb_began", None)
    if began is not None:
      began(count, line, verb, values)
    life._say(f"PROCEDURE {proc.name} step {count} (line {line}): "
              f"{st.Step(verb, values).describe()}")
    if bad:
      verdict = {"ok": False, "reason": "; ".join(bad)}
    else:
      verdict = yield from st.run_verb(life, spec, values,
                                       {"procedure": proc.name, "n": count,
                                        "line": line},
                                       until=t0 + proc.budget_s)
    entry = {"i": count - 1, "verb": verb, "line": line, **verdict,
             **({"tested": True} if tested else {})}
    result["steps"].append(entry)
    del result["inFlight"]
    if verdict.get("stopped") == "interrupted":
      # the robot's own interrupt ended a walk inside it (issue #381,
      # `run_verb`): an abort, as between two verbs, never a failed step
      raise _Stop("interrupted")
    if not verdict.get("ok") and tested:
      life._say(f"PROCEDURE {proc.name} step {count} (line {line}, {verb}) did not do "
                "what it said, as its test read"
                + (f" -- {verdict['reason']}" if verdict.get("reason") else ""))
      return False
    if not verdict.get("ok"):
      result["failedAt"] = count - 1
      life._say(f"PROCEDURE {proc.name} failed at step {count} (line {line}, {verb})"
                + (f" -- {verdict['reason']}" if verdict.get("reason") else ""),
                detail=verdict.get("trace", ""))
      raise _Stop("failed")
    result["completed"] = count
    return True

  def block(stmts) -> Routine:
    for s in stmts:
      kind = s[0]
      if kind == "verb":
        yield from verb_step(s[1], s[2], s[3])
      elif kind == "set":
        env[s[1]] = _eval(s[2], env, life, facts)
      elif kind == "if":
        for cond, body in s[1]:
          if cond[0] == "try":
            _, neg, (_, verb, args, line) = cond
            held = (yield from verb_step(verb, args, line, tested=True)) != neg
          else:
            held = _truth(_eval(cond, env, life, facts))
          if held:
            yield from block(body)
            break
        else:
          yield from block(s[2])
      elif kind == "repeat":
        for i in range(s[2]):
          env[s[1]] = float(i)
          yield from block(s[3])
      elif kind == "while":
        n = 0
        while _truth(_eval(s[1], env, life, facts)):
          n += 1
          if n > MAX_ITER:
            raise _Stop("loop-cap")
          yield from block(s[2])
      elif kind == "return":
        raise _Return()
      # "pass": nothing

  try:
    try:
      yield from block(proc.body)
      result["ok"] = True
    except _Return:
      result["ok"] = True
    except _Stop as stop:
      if stop.why != "failed":
        result["stopped"] = stop.why
        if stop.why not in ("interrupted", "budget", "steps"):
          # an evaluation fault is a failed step of its own, so the record
          # says where the arithmetic went wrong
          result["total"] = count + 1
          result["failedAt"] = count
          result["steps"].append({"i": count, "verb": "expr", "ok": False,
                                  "reason": stop.why})
          life._say(f"PROCEDURE {proc.name} stopped: {stop.why}")
  finally:
    # ...whatever ended it, a close from outside included (no yield here)
    result["seconds"] = round(float(life.data.time) - t0, 2)
    # The locals as they stood when the run ended: a procedure's variables
    # are its readout (issue #227) -- what it read and computed reaches the
    # mind through these, and through nothing else.
    result["locals"] = {k: round(float(v), 6) for k, v in env.items()}
  return result
