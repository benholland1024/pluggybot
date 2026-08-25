"""When work appears, how long it stands, and how much of it there may be
(issue #23).

hub/tasks.py owns what a task IS and what may happen to it, and says in its
own docstring that timing policy is deliberately not there. This is that
policy, and it is DATA -- hub/cadence.json, overridable per deploy with
$PLUGGY_CADENCE, on exactly the terms hub/rewards.json is data. The three
files divide cleanly and that is the point: what a job is, what it pays, and
when it turns up are re-tuned one at a time.

Three things this module exists to prevent, in the order they bite.

  AN EMPTY WORLD. `lifecycle.seed_tasks` put up a starter set once, at
  mission start, and nothing ever added another -- so a robot that worked
  through the board or watched it lapse spent the rest of a multi-hour run
  with nothing asked of it. A world that only ever offers work in its first
  second is a demo, not a world.

  A SPAMMED ONE. The other failure is louder and arrives by the same door:
  an offer rate set against wall-clock intuition rather than against the
  mission's own clock. A home charge-to-charge cycle is ~365 sim-seconds and
  ONE ERRAND COSTS ROUGHLY ONE FULL PACK, so the robot finishes about one job
  per cycle. Offer faster than the board's cap and the surplus lapses, which
  is honest; offer faster than that and the board is a wall of `expired`
  markers with a robot behind it that never had a chance.

  THE SAME WHITEBOARD, FOREVER. A rotation that picks the first buildable job
  every time books one target and ignores the rest. Two rules answer it, and
  both are about the TARGET rather than the kind: a target carrying an open
  task is never re-offered, and a target rests for `cooldownS` after being
  named. What is left is picked least-recently-offered-first, so a world's
  furniture gets used evenly.

⚠ DETERMINISTIC, and it has to be. There is no RNG here: the kind rotates on
a counter and the target is the least-recently-offered eligible one, ties
broken by name. A world that offered a different job every run would make
every mission test a different test -- the same argument `QuestionBank.pick`
and `overseer._fill` are built on.

⚠ ONE OFFER PER TICK, AND NO CATCH-UP. A tick that cannot place a job --
the board is capped, every target is booked or resting, or the pack cannot
fund the cheapest candidate -- waits for the next tick rather than
accumulating credit. Catching up in a burst is how a robot that spent twenty
minutes on one errand walks back into a board with eight new jobs on it,
which is the unbounded backlog this module is supposed to be preventing,
arriving through the front door instead of the back.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from pluggybot.hub.tasks import KINDS, TaskBoard

#: The shipped cadence. Overridable per deploy with $PLUGGY_CADENCE, which is
#: how the deployment re-tunes how busy the world is on a mounted volume
#: without a rebuild -- the same door hub/rewards.json and hub/questions.json
#: are opened by.
CADENCE_PATH = Path(__file__).with_name("cadence.json")
CADENCE_ENV = "PLUGGY_CADENCE"
CADENCE_VERSION = 1

#: Sim seconds between producer ticks in the mission loop. Not a cadence --
#: a POLLING interval, the granularity at which `everyS` and a deadline are
#: noticed. It exists because the tick hangs off the per-step seam (~500 Hz)
#: and sweeping a board of forty tasks that often is Python spent to learn
#: nothing: a job that lapses is a job nobody was going to get to in the next
#: second either.
CHECK_S = 1.0


@dataclass(frozen=True)
class Cadence:
  """One world's timing policy, as read off hub/cadence.json.

  Frozen, and every field is a number or a name -- a Cadence is the ANSWER to
  "how busy is this world", not a thing that decides it. `TaskProducer` below
  is what acts on one.
  """

  world: str = ""
  first_at_s: float = 300.0
  every_s: float = 300.0
  ttl_s: float | None = 600.0
  cooldown_s: float = 600.0
  max_offered: int = 5
  max_tasks: int = 40
  initial: int = 3
  #: kind name -> how it is built, IN ROTATION ORDER. A dict rather than a
  #: list because the params travel with the kind, and because JSON preserves
  #: the order they were written in -- which is the rotation.
  kinds: dict[str, dict] = field(default_factory=dict)
  path: Path | None = None

  @classmethod
  def load(cls, world: str = "", path: str | os.PathLike | None = None) -> "Cadence":
    """Read one world's block. `path`, else $PLUGGY_CADENCE, else the shipped
    file; the world's block overrides the default block key by key."""
    target = Path(path or os.environ.get(CADENCE_ENV) or CADENCE_PATH)
    doc = json.loads(target.read_text())
    version = int(doc.get("version", 0))
    if version != CADENCE_VERSION:
      # No backward-compatible read, exactly as for the reward table and the
      # question bank: this is a small hand-edited document, and running a
      # world's whole work schedule out of a shape this build does not
      # understand is worse than refusing to start.
      raise ValueError(f"{target}: cadence version {version}, "
                       f"expected {CADENCE_VERSION}")
    spec = dict(doc.get("default") or {})
    spec.update(dict((doc.get("worlds") or {}).get(world) or {}))
    return cls._build(world, spec, target)

  @classmethod
  def _build(cls, world: str, spec: dict, path: Path | None) -> "Cadence":
    kinds = {str(name): dict(cfg or {})
             for name, cfg in (spec.get("kinds") or {}).items()}
    unknown = [name for name in kinds if name not in KINDS]
    if unknown:
      # Refused rather than skipped, and this is the same lock `Task.create`
      # carries one door up: a cadence naming a kind that does not exist is a
      # world scheduled to offer work nothing can build or judge, and finding
      # that out at load is worth more than a board that is quietly emptier
      # than the file says.
      raise ValueError(
        f"{path}: cadence for {world or 'default'} offers unknown task "
        f"kind(s) {', '.join(sorted(unknown))} "
        f"(have: {', '.join(sorted(KINDS))})")
    out = cls(
      world=world,
      first_at_s=float(spec.get("firstAtS", 300.0)),
      every_s=float(spec.get("everyS", 300.0)),
      ttl_s=(None if spec.get("ttlS") is None else float(spec["ttlS"])),
      cooldown_s=float(spec.get("cooldownS", 600.0)),
      max_offered=int(spec.get("maxOffered", 5)),
      max_tasks=int(spec.get("maxTasks", 40)),
      initial=int(spec.get("initial", 3)),
      kinds=kinds, path=path)
    if out.every_s <= 0.0:
      raise ValueError(f"{path}: everyS must be positive, got {out.every_s}")
    if out.ttl_s is not None and out.ttl_s <= 0.0:
      raise ValueError(f"{path}: ttlS must be positive or null, "
                       f"got {out.ttl_s}")
    if out.max_offered < 1 or out.max_tasks < 1:
      raise ValueError(f"{path}: maxOffered and maxTasks must be at least 1 "
                       f"(got {out.max_offered} and {out.max_tasks})")
    if out.max_offered > out.max_tasks:
      raise ValueError(f"{path}: maxOffered {out.max_offered} exceeds "
                       f"maxTasks {out.max_tasks} -- the board could never "
                       "hold the offers it is allowed to stand")
    if out.initial < 0:
      raise ValueError(f"{path}: initial must not be negative")
    return out


_cache: dict[str, Cadence] = {}


def default_cadence(world: str = "") -> Cadence:
  """The process-wide cadence for a world, loaded once."""
  if world not in _cache:
    _cache[world] = Cadence.load(world)
  return _cache[world]


def reload_cadence(path: str | os.PathLike | None = None,
                   world: str = "") -> Cadence:
  _cache.pop(world, None)
  loaded = Cadence.load(world, path)
  _cache[world] = loaded
  return loaded


class TaskProducer:
  """The thing that puts jobs into a world, on a schedule.

  Deliberately knows nothing about the world it is producing for beyond
  `targets`: a mapping of `TaskKind.target_kind` ("board", "zone", "module")
  to the names this world actually has. That is what keeps the layering one
  way -- hub/lifecycle.py knows about worlds and hands the furniture in, and
  nothing here imports it back.

  `board` is the `TaskBoard`, which stays the only thing that moves a task.
  The producer offers; expiry, claiming, resolution and persistence are all
  still the board's, and every offer made here goes out as an ordinary
  `task_offered` message on the board's own hooks.
  """

  def __init__(self, board: TaskBoard, cadence: Cadence,
               targets: dict[str, list[str]] | None = None) -> None:
    self.board = board
    self.cadence = cadence
    self.targets = {kind: list(names)
                    for kind, names in (targets or {}).items() if names}
    # Only the kinds this world has something to point at. A world with no
    # whiteboards offers fewer jobs rather than offering one nothing can
    # build -- the rule `seed_tasks` had and the reason it read the book.
    self.kinds = tuple(name for name in cadence.kinds
                       if name in KINDS
                       and KINDS[name].target_kind in self.targets)
    #: rotation cursor over `self.kinds`
    self.cursor = 0
    #: ...and over each kind's `programs` list, so successive drawings differ
    self.figure = 0
    #: target -> sim time it was last named by an offer
    self.last_offered: dict[str, float] = {}
    #: sim time of the next tick that may place an offer
    self.next_at = float(cadence.first_at_s)
    self.made = 0
    self.deferred = 0

  # ---- the two ways a job gets made ----------------------------------------

  def seed(self, t: float = 0.0, pack_wh: float | None = None) -> list:
    """Put up the starting board, once, at mission start.

    Up to `initial` offers -- fewer if the world runs out of unbooked
    targets, which is the ordinary case: home has two whiteboards and three
    board-shaped kinds, so one of them waits for a board to come free. That
    is the per-target rule doing its job at t=0 rather than an accident.

    ⚠ Call this AFTER every hook is attached. `TaskBoard.offer` emits a
    `task_offered` the moment it is called, so seeding at construction time
    puts the offers on the floor before the recorder exists -- a recording
    whose tasks block is populated and whose offer events are missing.
    """
    made = []
    for _ in range(self.cadence.initial):
      task = self._offer_one(t, pack_wh)
      if task is None:
        break
      made.append(task)
    self.next_at = float(t) + self.cadence.first_at_s
    return made

  def tick(self, t: float, pack_wh: float | None = None) -> list:
    """Place at most one offer, if one is due. Returns what was offered.

    Cheap enough to call on the physics seam: until `next_at` this is one
    float compare.
    """
    if float(t) < self.next_at:
      return []
    task = self._offer_one(t, pack_wh)
    # Advanced whether or not anything was placed, and without catch-up: see
    # the module docstring. A tick that could not place a job is a tick the
    # world was full, not a debt the world owes the robot.
    self.next_at = float(t) + self.cadence.every_s
    return [task] if task is not None else []

  # ---- picking one ---------------------------------------------------------

  def _offer_one(self, t: float, pack_wh: float | None):
    if not self.kinds:
      return None
    if len(self.board.offered()) >= self.cadence.max_offered:
      return None
    booked = {task.target for task in self.board.open_tasks()}
    n = len(self.kinds)
    starved = False
    # ⚠ WHERE THE HEAD ENDS UP IS THE WHOLE OF FAIRNESS HERE. A kind that was
    # passed over KEEPS ITS PLACE at the front of the queue; only a kind that
    # was actually offered gives its turn up. Without that, a kind can be
    # starved forever by the ones in front of it -- home has three
    # board-shaped kinds and two whiteboards, and with a cursor that simply
    # advanced past whatever it placed, `rate_artwork` (the third one, and
    # the entire visitor-rated tier) was offered ZERO times in four
    # sim-hours, measured. The two in front booked both boards every cycle
    # and the census saved the attempt, so the rotation never reached it.
    passed: int | None = None
    for step in range(n):
      index = (self.cursor + step) % n
      kind = self.kinds[index]
      spec = KINDS[kind]
      # ⚠ THE TARGET IS PICKED BEFORE THE ENERGY GATE (issue #15), because
      # what a job costs depends on WHICH whiteboard it names: home's far one
      # is 7 m away through a doorway and measures 0.14 Wh more than its near
      # one. Gating the kind first would refuse `draw_figure` outright in a
      # world that can draw on one of its two boards perfectly well.
      target = self._pick_target(spec.target_kind, booked, t)
      if target is None:
        passed = index if passed is None else passed
        continue
      # Priced by the BOARD, which knows this world's measured costs;
      # `spec.estimate_wh` is the fallback for a world nobody has measured. A
      # world-agnostic figure here refused room_hub every `fetch_module` it
      # can do perfectly well, because home's carry is 0.12 Wh dearer and one
      # number cannot be both.
      estimate = self.board.estimate_for(kind, target)
      if estimate is None:
        estimate = spec.estimate_wh
      if pack_wh is not None and estimate > float(pack_wh):
        # THE WORLD'S ENERGY GATE, and read what it is measured against.
        #
        # `pack_wh` here is what a CHARGED pack holds in this world, not what
        # is in the cell at this instant -- the caller passes
        # `HubLifecycle.fundable_wh`. So this refuses a job the world could
        # never fund and nothing else, and a job the robot merely cannot
        # afford right now is still put up.
        #
        # ⚠ THE OTHER READING WAS TRIED AND MEASURED, and it starves the
        # robot. Issue #23 asks for a job whose cost exceeds the usable pack
        # to be deferred until after a charge; gating each tick on the
        # INSTANTANEOUS charge does that literally and takes home from 58
        # offers in four sim-hours to 14, with 46 deferrals -- fewer jobs
        # than the robot can complete in that time, which is the empty world
        # this module exists to prevent, arriving as a safety feature. The
        # reason is arithmetic rather than tuning: one errand costs roughly
        # one full pack, so the window in which a home cell is above any
        # errand's estimate is the minute after a charge, and a 240 s tick
        # mostly misses it.
        #
        # Deferring until after a charge is REAL, and it already lives in
        # `Task.claimable` -- an offer simply stands, unclaimable, until the
        # pack is legal again, which is also what the site draws and what
        # `_claim_next_task` consults. That is the gate with the teeth; this
        # one only keeps a world from advertising work it could never fund.
        starved = True
        passed = index if passed is None else passed
        continue
      self.cursor = passed if passed is not None else (index + 1) % n
      return self._offer(kind, target, t)
    # Counted once per attempt, not once per candidate skipped: what a long
    # run wants to know is how many offers the battery cost it, not how many
    # comparisons were made.
    self.deferred += 1 if starved else 0
    # The cursor is deliberately left where it is: nothing was placed, so
    # nobody took a turn.
    return None

  def _pick_target(self, target_kind: str, booked: set, t: float) -> str | None:
    """The least-recently-offered target of this kind that is free to take a
    job. `None` if every one of them is booked or resting.

    Least-recently-offered rather than first: with two whiteboards and three
    board-shaped kinds, "first" books whiteboard_a forever and the second
    board is scenery.
    """
    ready = [name for name in self.targets.get(target_kind, ())
             if name not in booked
             and float(t) - self.last_offered.get(name, float("-inf"))
             >= self.cadence.cooldown_s]
    if not ready:
      return None
    return min(ready, key=lambda name: (self.last_offered.get(name, float("-inf")),
                                        name))

  def _offer(self, kind: str, target: str, t: float):
    params, secret = self._build(kind, target)
    task = self.board.offer(kind, target, params=params, secret=secret,
                            ttl=self.cadence.ttl_s, t=t)
    if task is None:
      return None
    self.last_offered[target] = float(t)
    self.made += 1
    return task

  def _build(self, kind: str, target: str) -> tuple[dict, dict]:
    """The params and the secret one offer is made with.

    The only kind with a secret is `whiteboard_answer` (issue #22), and its
    question is drawn off the BOARD's own sequence number -- which survives a
    restart, so a deployed robot works through the bank instead of being
    asked the same thing every morning, while a test seeding a fresh board
    gets the same question every time.
    """
    cfg = self.cadence.kinds.get(kind) or {}
    params: dict = dict(cfg.get("params") or {})
    programs = list(cfg.get("programs") or ())
    if programs:
      params["program"] = programs[self.figure % len(programs)]
      self.figure += 1
    if kind == "whiteboard_answer":
      from pluggybot.hub.questions import default_bank
      question = default_bank().pick(self.board.seq)
      params.update({"question": question.ask, "template": question.id})
      return params, {"answer": question.answer}
    return params, {}

  # ---- what a long run looks like ------------------------------------------

  def stats(self) -> dict:
    return {"offered": self.made, "deferred": self.deferred,
            "nextAt": round(self.next_at, 1),
            "kinds": list(self.kinds)}
