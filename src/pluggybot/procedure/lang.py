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
            look()
            if read("look.tag") >= 0 and read("look.range") < 1.0:
                wait(2)
        while read("arm") < 0.05 and n < 10:  # capped at MAX_ITER iterations
            move("arm", read("arm") + 0.01)   # the motor level: a ramped setpoint
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
verb (`move` is `HubSwap.ramp_routine`), the scoring path not at all -- a
`Verdict` is sealed -- and nothing here writes `data.ctrl`.
"""

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Any

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
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
#: Names a local may not shadow: the language's own words and every verb.
RESERVED = frozenset({"read", "budget", "range", "def", "and", "or", "not",
                      "if", "elif", "else", "for", "while", "in", "return",
                      "pass", "True", "False", "None"})

_BIN = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
_CMP = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
        ast.Eq: "==", ast.NotEq: "!="}


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
          for _, body in s[1]:
            found = walk(body)
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
  def __init__(self, source: str) -> None:
    self.source = source
    self.reasons: list[str] = []

  def refuse(self, node, what: str) -> None:
    line = getattr(node, "lineno", 0)
    self.reasons.append(f"line {line}: {what}")

  # -- expressions --
  def expr(self, node) -> Any:
    if isinstance(node, ast.Constant):
      if isinstance(node.value, bool):
        return ("num", 1.0 if node.value else 0.0)
      if isinstance(node.value, (int, float)) and math.isfinite(node.value):
        return ("num", float(node.value))
      self.refuse(node, f"a {type(node.value).__name__} is not a number")
      return ("num", 0.0)
    if isinstance(node, ast.Name):
      if not NAME_RE.match(node.id) or node.id in RESERVED or node.id in st.VERBS:
        self.refuse(node, f"{node.id!r} is not a name this language allows")
      return ("var", node.id)
    if isinstance(node, ast.Call):
      fn = node.func
      if isinstance(fn, ast.Name) and fn.id == "read":
        if (len(node.args) != 1 or node.keywords
            or not isinstance(node.args[0], ast.Constant)
            or not isinstance(node.args[0].value, str)):
          self.refuse(node, "read takes one sensor name in quotes")
          return ("num", 0.0)
        return ("read", node.args[0].value)
      self.refuse(node, "only read(...) may be used inside an expression; "
                        "a verb is a statement of its own")
      return ("num", 0.0)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
      return ("bin", _BIN[type(node.op)], self.expr(node.left), self.expr(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
      return ("neg", self.expr(node.operand))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
      return ("not", self.expr(node.operand))
    if isinstance(node, ast.Compare):
      if len(node.ops) != 1 or type(node.ops[0]) not in _CMP:
        self.refuse(node, "compare two things with one of < <= > >= == !=")
        return ("num", 0.0)
      return ("cmp", _CMP[type(node.ops[0])], self.expr(node.left),
              self.expr(node.comparators[0]))
    if isinstance(node, ast.BoolOp):
      op = "and" if isinstance(node.op, ast.And) else "or"
      out = self.expr(node.values[0])
      for v in node.values[1:]:
        out = (op, out, self.expr(v))
      return out
    self.refuse(node, f"{type(node).__name__} is not part of this language")
    return ("num", 0.0)

  def arg(self, node) -> Any:
    """A verb argument: a quoted name, or a numeric expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
      return ("str", node.value)
    return self.expr(node)

  # -- statements --
  def block(self, nodes, depth: int) -> tuple:
    return tuple(self.stmt(n, depth) for n in nodes)

  def stmt(self, node, depth: int) -> Any:
    if depth > MAX_NESTING:
      self.refuse(node, f"nested deeper than {MAX_NESTING}")
      return ("pass",)
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
      call = node.value
      fn = call.func
      if not isinstance(fn, ast.Name):
        self.refuse(node, "a statement is a verb call by its bare name")
        return ("pass",)
      if fn.id == "read":
        self.refuse(node, "read(...) is an expression; assign or compare it")
        return ("pass",)
      verb = st.VERBS.get(fn.id)
      if verb is None:
        self.refuse(node, f"unknown verb {fn.id!r} (have: {', '.join(st.VERBS)})")
        return ("pass",)
      names = list(verb.args)
      args: dict = {}
      if len(call.args) > len(names):
        self.refuse(node, f"{fn.id} takes {len(names)} argument(s)")
      for name, value in zip(names, call.args):
        args[name] = self.arg(value)
      for kw in call.keywords:
        if kw.arg is None or kw.arg not in names:
          self.refuse(node, f"{fn.id} takes no {kw.arg or '**'}")
          continue
        args[kw.arg] = self.arg(kw.value)
      missing = [n for n in names if n not in args]
      if missing:
        self.refuse(node, f"{fn.id} needs {', '.join(missing)}")
      return ("verb", fn.id, args, getattr(node, "lineno", 0))
    if isinstance(node, ast.Assign):
      if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        self.refuse(node, "assign to one plain name")
        return ("pass",)
      name = node.targets[0].id
      if not NAME_RE.match(name) or name in RESERVED or name in st.VERBS:
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
      arms = [(self.expr(node.test), self.block(node.body, depth + 1))]
      orelse = node.orelse
      while len(orelse) == 1 and isinstance(orelse[0], ast.If):
        arms.append((self.expr(orelse[0].test),
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


def parse(source: str) -> Procedure:
  """Surface syntax -> Procedure, or Refused with every reason."""
  proc, reasons = _parse(source)
  if reasons:
    raise Refused(reasons)
  return proc


def _parse(source: str) -> tuple:
  """The parse with its reasons, so `compile_procedure` can add the
  world's on top and refuse everything at once."""
  if len(source) > MAX_SOURCE_CHARS:
    raise Refused([f"the source is {len(source)} characters; the cap is "
                   f"{MAX_SOURCE_CHARS}"])
  try:
    tree = ast.parse(source)
  except SyntaxError as e:
    raise Refused([f"line {e.lineno or 0}: {e.msg}"]) from None
  if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
    raise Refused(["a procedure is exactly one `def name():`"])
  fn = tree.body[0]
  p = _Parser(source)
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
    if call.args:
      p.refuse(call, "budget takes keywords: budget(steps=N, seconds=S)")
    for kw in call.keywords:
      lit = kw.value.value if isinstance(kw.value, ast.Constant) else None
      if kw.arg == "steps" and isinstance(lit, int) and not isinstance(lit, bool):
        steps_budget = lit
      elif kw.arg == "seconds" and isinstance(lit, (int, float)) and not isinstance(lit, bool):
        budget_s = float(lit)
      else:
        p.refuse(call, f"budget({kw.arg}=...) is not a literal steps or seconds")
    if not 1 <= steps_budget <= MAX_STEPS:
      p.refuse(call, f"steps={steps_budget} is outside 1..{MAX_STEPS}")
    if not 0.0 < budget_s <= MAX_BUDGET_S:
      p.refuse(call, f"seconds={budget_s} is outside (0, {MAX_BUDGET_S:.0f}]")
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
      n += sum(_count_verbs(b) for _, b in s[1]) + _count_verbs(s[2])
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
        spec = st.VERBS[verb]
        lit = {}
        for name, value in args.items():
          arg = spec.args[name]
          if value[0] == "str":
            if arg.kind != "str":
              bad.append(f"line {line}: {verb}({name}=...) takes a number, "
                         "not a name in quotes")
            else:
              lit[name] = value[1]
          elif arg.kind == "str":
            bad.append(f"line {line}: {verb}({name}=...) takes a name in quotes")
          elif value[0] == "num":
            lit[name] = value[1]
          else:
            _sensors(value, bad)
        bad.extend(f"line {line}: {r}"
                   for r in st.check_step(spec, lit, facts, partial=True))
      elif s[0] == "if":
        for cond, body in s[1]:
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

  def _sensors(expr, out):
    if not isinstance(expr, tuple):
      return
    if expr[0] == "read" and expr[1] not in facts.sensors:
      out.append(f"unknown sensor {expr[1]!r} (have: {', '.join(facts.sensors)})")
    for sub in expr[1:]:
      if isinstance(sub, tuple):
        _sensors(sub, out)

  walk(proc.body)
  return list(dict.fromkeys(bad))


def compile_procedure(source: str, facts: WorldFacts) -> Procedure:
  """Parse and validate, and refuse with EVERY reason from both."""
  proc, reasons = _parse(source)
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
    return float(sensor.read(life))
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


def run_procedure_routine(life, proc: Procedure, facts: WorldFacts) -> Routine:
  """Interpret a validated procedure as a routine. The result has the shape a
  program's run has (steps.run_program_routine), so the errand around it and
  the evaluator read both alike: one verdict per verb executed, honest about
  how far it got and why it stopped."""
  reasons = validate(proc, facts)
  if reasons:
    raise Refused(reasons)
  result: dict[str, Any] = {"program": proc.name, "role": st.DEFAULT_ROLE,
                            "total": 0, "completed": 0, "steps": [],
                            "ok": False, "source": proc.source}
  env: dict[str, float] = {}
  t0 = float(life.data.time)
  count = 0

  def verb_step(verb: str, args: dict, line: int) -> Routine:
    nonlocal count
    if count and life.interrupted():
      raise _Stop("interrupted")
    if float(life.data.time) - t0 > proc.budget_s:
      raise _Stop("budget")
    if count >= proc.steps_budget:
      raise _Stop("steps")
    spec = st.VERBS[verb]
    values: dict = {}
    for name, value in args.items():
      values[name] = value[1] if value[0] == "str" else _eval(value, env, life, facts)
    # a computed argument is checked here, against the same rules a literal
    # met at compile time
    bad = st.check_step(spec, values, facts)
    count += 1
    result["total"] = count
    life._say(f"PROCEDURE {proc.name} step {count} (line {line}): "
              f"{st.Step(verb, values).describe()}")
    if bad:
      verdict = {"ok": False, "reason": "; ".join(bad)}
    else:
      verdict = yield from spec.run(life, values)
    entry = {"i": count - 1, "verb": verb, "line": line, **verdict}
    result["steps"].append(entry)
    if not verdict.get("ok"):
      result["failedAt"] = count - 1
      life._say(f"PROCEDURE {proc.name} failed at step {count} (line {line})"
                + (f" -- {verdict['reason']}" if verdict.get("reason") else ""))
      raise _Stop("failed")
    result["completed"] = count

  def block(stmts) -> Routine:
    for s in stmts:
      kind = s[0]
      if kind == "verb":
        yield from verb_step(s[1], s[2], s[3])
      elif kind == "set":
        env[s[1]] = _eval(s[2], env, life, facts)
      elif kind == "if":
        for cond, body in s[1]:
          if _truth(_eval(cond, env, life, facts)):
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
  result["seconds"] = round(float(life.data.time) - t0, 2)
  return result
