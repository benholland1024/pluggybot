"""Seeded random days on the stub for #730 (a robot answers while it charges):
messages from people and from the other robot, ratings and ticket lines,
answers of every action and every failure kind, slow calls that outlive
their deadline, `message_received`, periodic and failure rows, charges at
random packs, deaths, stand-ups and the run's hour -- one robot or a pair
on one physics loop -- with the invariants Saturday's wake needs read
after every step (`Watch`):

  1. never unplugged below `CHARGED` but by an answer or a row of the
     agent's own that moves the body, or a charge's own end;
  2. the clock that kills a mind nobody consults never older than the last
     ask a break made, and no `unminded` death within `UNMINDED_AFTER_S`
     of one;
  3. a plug-in episode banks at most one `charge` row and one cycle, its
     `task_complete charge` once a cycle;
  4. a person's message caused at most one failed call (N = 1), and none
     is lost: answered, shown in a call, or still waiting;
  5. no call is made for a message the model answered, and none for one
     an answer to nobody has met `OWED_SHOWINGS` times.

Not collected (no `test_` prefix): `test_charge_answer.py` flies a cheap
seeded slice of it, and the loop's runner flies thousands as a
measurement. The calls are made on the SIM's clock -- an answer drawn as
the ask is made, ready a drawn think later, a slow one held past its
deadline until a drawn moment -- so a seed is one day, bit for bit.
"""
from __future__ import annotations

import math
import random
import threading
import time
from types import SimpleNamespace

import mujoco

from pluggybot import lifecycle as lc
from pluggybot import tick
from pluggybot.body import STUB_WORLD, RackPose, StubBody
from pluggybot.economy import energy
from pluggybot.economy.ledger import Account
from pluggybot.mind import events as ev
from pluggybot.mind import overseer as ov
from pluggybot.mind import text as text_registry
from pluggybot.mind.inbox import Inbox
from pluggybot.robot import FIRST, SECOND

from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path
from test_overseer import CutResponse, FakeResponse, full

WORLD = "home_humanoid"
#: The stub's world at a coarser step: what the loop decides is in seconds,
#: and a day is ten times quicker to fly.
FAST_WORLD = STUB_WORLD.replace('<mujoco model="stub">',
                                '<mujoco model="stub"><option timestep="0.02"/>')
FAILURES = ("timeout", "offline", "garbled", "cutoff", "slow")


class Violation(AssertionError):
  """An invariant the wake needs, broken: what, when, and the day's seed."""


class _Mode:
  """The operator's switch, flipped by the day's schedule."""

  def __init__(self) -> None:
    self.mode, self.at, self.errors = "llm", None, []

  def poll(self, force: bool = False) -> str:
    return self.mode

  @property
  def paused(self) -> bool:
    return False

  @property
  def thinking(self) -> bool:
    return self.mode == "llm"

  def snapshot(self) -> dict:
    return {"mode": self.mode, "notes": []}


class _Client:
  """`client.messages.create`, answered off the mind's drawn plan."""

  def __init__(self, mind: "SimMind") -> None:
    self.mind = mind
    self.messages = self

  def create(self, **kwargs):
    return self.mind.answer()


class SimMind(ov.Overseer):
  """The overseer, its calls made on the sim's clock. An answer is drawn as
  the ask is made: a decision ready a drawn think later, or a failure of
  each kind -- a slow one held past its deadline (the next ask finds it out
  `busy`) until the day's hook lets it go at a drawn moment."""

  def __init__(self, menu, rng: random.Random, life_box: list, **kw) -> None:
    super().__init__(menu, **kw)
    self.rng, self.life_box = rng, life_box
    self._client, self._client_ready = _Client(self), True
    self.hold_until = 0.0
    self.plan: tuple = ("ok", {})
    self.slow: list = []                  # [(release_t, event, thread)]
    self.calls_made: list[dict] = []
    self.fail_share = rng.choice((0.0, 0.2, 0.5, 0.9))

  @property
  def life(self):
    return self.life_box[0]

  @property
  def pending(self) -> bool:
    return self.clock() < self.hold_until or super().pending

  def _draw(self, state: dict) -> tuple:
    rng, now = self.rng, self.clock()
    if rng.random() < self.fail_share:
      kind = rng.choice(FAILURES)
      if kind == "slow":
        return ("slow", now + self.timeout_s + ov.POLL_GRACE_S + rng.uniform(1.0, 120.0))
      return (kind,)
    action = rng.choice(("charge", "charge", "idle", "idle", "explore"))
    answer = full(action=action, reason="fuzz")
    shown = [m.get("id") for m in state.get("visitorMessages", ()) if m.get("id")]
    if shown and rng.random() < 0.7:
      answer.update(respond_to=rng.choice(shown), outcome="replied", reply="hello")
    if rng.random() < 0.05:                # the agent edits its rule for a message
      answer["event_map"] = [{"event": "message_received",
                              "action": rng.choice(("remove", "ask", "idle"))}]
    return ("ok", answer)

  def start(self, state: dict) -> bool:
    self.plan = self._draw(state)
    self.hold_until = (self.clock() + self.rng.choice((0.0, 0.5, 5.0, 30.0, 90.0))
                       if self.plan[0] != "slow" else 0.0)
    real = ov.threading
    ov.threading = SimpleNamespace(Thread=self._thread, Lock=real.Lock, RLock=real.RLock,
                                   Event=real.Event)
    self._deferred = None
    try:
      called = super().start(state)
    finally:
      ov.threading = real
    if self._deferred is not None:
      # ...run once `start` has let its lock go, which the call takes too
      target, args = self._deferred
      self._deferred = None
      target(*args)
    return called

  def _thread(self, target, args=(), daemon=True):
    """`start`'s worker: run as `start` returns, or held for a slow answer."""
    plan = self.plan

    class _Run:
      def start(inner) -> None:
        self.calls_made.append({"t": self.clock(), "plan": plan[0],
                                "state": dict(self._pending_state)})
        if plan[0] != "slow":
          self._deferred = (target, args)
          return
        gate = threading.Event()
        worker = threading.Thread(target=target, args=args, daemon=True)
        self.slow.append((plan[1], gate, worker))
        self._gate = gate
        worker.start()
    return _Run()

  def release_due(self) -> None:
    """The day's hook: a slow call whose moment came answers now, late."""
    now = self.clock()
    for item in [s for s in self.slow if s[0] <= now]:
      _, gate, worker = item
      gate.set()
      worker.join(timeout=10.0)
      self.slow.remove(item)

  def answer(self):
    """What the drawn plan answers, or raises."""
    plan = self.plan
    if plan[0] == "slow":
      self._gate.wait(timeout=60.0)
      return FakeResponse(full(action="idle", reason="late"))
    if plan[0] == "timeout":
      raise TimeoutError("nobody answered in time")
    if plan[0] == "offline":
      raise ConnectionError("the endpoint is down")
    if plan[0] == "garbled":
      return FakeResponse("this is not a decision")
    if plan[0] == "cutoff":
      return CutResponse("")
    return FakeResponse(plan[1])


def _map(rng: random.Random) -> list:
  rows = []
  roll = rng.random()
  if roll < 0.8:
    rows.append(ev.Row(event="message_received", action=ev.ASK))
  elif roll < 0.9:
    rows.append(ev.Row(event="message_received", action="idle"))
  if rng.random() < 0.5:
    rows.append(ev.Row(event="every", action=ev.ASK, value=float(rng.choice((300, 900, 1700)))))
  if rng.random() < 0.6:
    rows.append(ev.Row(event="decision_failed", action=rng.choice(("explore", "idle", "charge"))))
  if rng.random() < 0.5:
    rows.append(ev.Row(event="task_complete", kind="charge", action=ev.ASK))
  if rng.random() < 0.5:
    rows.append(ev.Row(event="nothing_to_do", action=ev.ASK))
  if rng.random() < 0.5:
    rows.append(ev.Row(event="battery_below", value=rng.choice((0.3, 0.5)), action="charge"))
  rng.shuffle(rows)
  return rows


class Watch:
  """One robot's invariants, read after every step and at its spies."""

  def __init__(self, life, mind: SimMind, seed: int) -> None:
    self.life, self.mind, self.seed = life, mind, seed
    self.last_break_t: float | None = None
    self.acting: list[str] = []
    self.completed_t: float | None = None
    self.episode: dict | None = None
    self.episodes: list[dict] = []
    self.answered: set[str] = set()
    self.shown: set[str] = set()
    self.failed_for: dict[str, int] = {}
    self.asked_for: dict[str, int] = {}
    self.nobody_for: dict[str, int] = {}
    self.offered: set[str] = set()
    self.unplugs: list[tuple] = []
    self._wire()

  def fail(self, what: str) -> None:
    raise Violation(f"seed {self.seed} {self.life.root} t={float(self.life.data.time):.2f}: {what}")

  def _wire(self) -> None:
    life, body = self.life, self.life.body
    act = life._act_routine

    def acting(decision):
      self.acting.append(decision.source)
      try:
        return (yield from act(decision))
      finally:
        self.acting.pop()
    life._act_routine = acting
    complete = life._charge_complete

    def completing(before, said=""):
      self.completed_t = float(life.data.time)
      if self.episode is not None:
        self.episode["cycles"] += 1
      complete(before, said)
    life._charge_complete = completing
    bank = life._bank

    def banking(verdict):
      if verdict is not None and verdict.task == "charge" and self.episode is not None:
        self.episode["rows"] += 1
      return bank(verdict)
    life._bank = banking
    occur = life._occur

    def occurring(event, kind=""):
      if (event, kind) == ("task_complete", "charge") and self.episode is not None:
        self.episode["told"] += 1
      occur(event, kind)
    life._occur = occurring
    dock = body.dock_routine

    def docking():
      out = yield from dock()
      if body.on_charger and self.episode is None:
        self.episode = {"t": float(life.data.time), "rows": 0, "cycles": 0, "told": 0}
      return out
    body.dock_routine = docking

    def undock():
      frac = life.battery.fraction
      source = self.acting[-1] if self.acting else ""
      ended = self.completed_t == float(life.data.time)   # ...a charge's own end
      self.unplugs.append((float(life.data.time), round(frac, 4), source, ended))
      if (frac < lc.CHARGED - 1e-9 and not ended
          and not (source.startswith("llm") or source.startswith("event:"))):
        self.fail(f"unplugged at {frac:.3f} by {source or 'nothing acting'}")
      body.on_charger = False
      self._close_episode()
      return "arrived"
      yield
    body.undock_routine = undock
    stand = life._stand_up

    def standing(*a, **kw):
      stand(*a, **kw)
      body.on_charger = False             # ...a warp takes the plug out
      self._close_episode()
    life._stand_up = standing
    decide = life._decide_routine

    def deciding(asked_by=None, plugged=False):
      if plugged:
        self.last_break_t = float(life.data.time)
      return (yield from decide(asked_by, plugged))
    life._decide_routine = deciding
    asked = life._asked_about

    def asked_about(shown, newly, decision):
      owed = set(shown) & (set(life._owed) | set(newly))
      for msg_id in owed:
        if msg_id in self.answered:
          self.fail(f"asked again for {msg_id}, answered already")
        self.asked_for[msg_id] = self.asked_for.get(msg_id, 0) + 1
        if not decision.scripted and not decision.responds:
          # ...an answer to nobody counts against it: a mind that ignores a
          # message is asked about it no more than `OWED_SHOWINGS` times (an
          # answer to somebody else answers a message the queue then loses)
          self.nobody_for[msg_id] = self.nobody_for.get(msg_id, 0) + 1
          if self.nobody_for[msg_id] > lc.OWED_SHOWINGS:
            self.fail(f"{self.nobody_for[msg_id]} answers to nobody asked for {msg_id}")
        if decision.scripted:
          self.failed_for[msg_id] = self.failed_for.get(msg_id, 0) + 1
          if self.failed_for[msg_id] > 1:
            self.fail(f"{self.failed_for[msg_id]} failed calls for {msg_id} (N = 1)")
      self.shown |= set(shown)
      if not decision.scripted and decision.responds:
        self.answered.add(decision.respond_to)
      asked(shown, newly, decision)
    life._asked_about = asked_about

  def _close_episode(self) -> None:
    ep, self.episode = self.episode, None
    if ep is None:
      return
    self.episodes.append(ep)
    if ep["rows"] > 1 or ep["cycles"] > 1 or ep["told"] != ep["cycles"]:
      self.fail(f"one plug-in, {ep}")

  def step(self) -> None:
    life = self.life
    if self.last_break_t is not None and life._last_ask_t < self.last_break_t - 1e-9:
      self.fail(f"the death clock {life._last_ask_t} older than a break's ask {self.last_break_t}")
    if (life.dead is not None and life.dead.get("cause") == "unminded"
        and self.last_break_t is not None
        and life.dead["t"] - self.last_break_t < lc.UNMINDED_AFTER_S - 1.0):
      self.fail(f"unminded {life.dead['t'] - self.last_break_t:.0f} s after a break's ask")
    if self.episode is None and life.body.on_charger:
      self.episode = {"t": float(life.data.time), "rows": 0, "cycles": 0, "told": 0}

  def finish(self) -> None:
    self._close_episode()
    queued = {m.id for m in self.life.inbox.peek(len(self.life.inbox))}
    for msg_id in self.offered:
      if msg_id not in self.answered | self.shown | queued:
        self.fail(f"{msg_id} lost: never answered, shown or waiting")


def _robot(seed: int, model, data, handle, name: str, ledger):
  rng = random.Random(f"{seed}:{handle.root}")
  cfg = lc.world_config(WORLD)
  x, y, yaw = cfg["dock"]
  body = StubBody(model, data, handle=handle, rack=RackPose(x, y, yaw + math.pi),
                  grid_bounds=cfg["grid_bounds"], draw_w=rng.choice((10.0, 30.0, 60.0)),
                  charge_w=energy.load(WORLD).charge_w)
  menu = ov.Menu.for_world(WORLD, lc.board_book(WORLD))
  box: list = []
  mind = SimMind(menu, rng, box, origin="unseeded", event_map=ev.EventMap(_map(rng)),
                 robot_name=name, clock=lambda: float(data.time),
                 timeout_s=rng.choice((20.0, 60.0)))
  life = stub_life(WORLD, body=body, handle=handle, robot_name=name, overseer=mind,
                   inbox=Inbox(), ledger=ledger, mortal=True,
                   restart_after_s=rng.choice((60.0, 180.0)))
  box.append(life)
  life.home_pose = tuple(float(v) for v in cfg["start"])
  life.continuing = True
  life.mode = _Mode() if handle is FIRST else None
  life._minded = True
  life.battery.energy_wh = life.battery.capacity_wh * rng.uniform(0.15, 0.95)
  return life, mind, rng


def fly(seed: int, sim_s: float = 900.0, pair: bool = False) -> dict:
  """One seeded day; raises `Violation` at the first broken invariant."""
  from pluggybot.economy.ledger import Ledger
  model = mujoco.MjModel.from_xml_string(FAST_WORLD)
  data = mujoco.MjData(model)
  book = Ledger(robots=(FIRST.root, SECOND.root))
  handles = (FIRST, SECOND) if pair else (FIRST,)
  robots = [_robot(seed, model, data, h, n, Account(book, h.root))
            for h, n in zip(handles, ("Pluggy", "Rowan"))]
  lives = [r[0] for r in robots]
  if pair:
    a, b = lives
    a.peers, b.peers = [b], [a]
  watches = [Watch(life, mind, seed) for life, mind, _ in robots]
  sched = random.Random(f"{seed}:world")
  events = []
  t = 0.0
  while t < sim_s:
    t += sched.expovariate(1.0 / 40.0)
    events.append((t, sched.random()))
  first = lives[0]
  k = {"n": 0}

  def world() -> None:
    now = float(data.time)
    for life, mind, _ in robots:
      mind.release_due()
    while events and events[0][0] <= now:
      _, roll = events.pop(0)
      life = sched.choice(lives)
      k["n"] += 1
      n = k["n"]
      if roll < 0.55:
        msg_id = f"v{n}"
        life.inbox.offer({"type": "message", "id": msg_id, "from": "a visitor", "text": "hi"})
        watches[lives.index(life)].offered.add(msg_id)
      elif roll < 0.62:
        life.inbox.offer({"type": "message", "id": f"t{n}", "from": "Rowan", "text": "hello"},
                         sender=text_registry.PEER)
      elif roll < 0.68:
        life.inbox.offer({"type": "rating", "seq": 1, "quality": 0.5})
      elif roll < 0.72:
        life.inbox.offer({"type": "ticket_reply", "id": f"op{n}", "ticket": "tk_0001",
                          "text": "seen"})
      elif roll < 0.76:
        if life.dead is None:
          life._die("unpaid", "the fuzz's bill")
      elif roll < 0.79:
        life.inbox.offer({"type": "reset_robot", "from": "admin"})
      elif roll < 0.85 and first.mode is not None:
        first.mode.mode = "scripted" if first.mode.mode == "llm" else "llm"
      elif roll < 0.92:
        life.battery.energy_wh = max(0.0, life.battery.energy_wh - life.battery.capacity_wh * 0.2)
    for w in watches:
      w.step()
  first.body.step_hooks.append(world)
  cfg = lc.world_config(WORLD)
  starts = (cfg["start"], cfg["start2"])
  wall0 = time.monotonic()
  if pair:
    days = [life.begin(s, max_sim_time=sim_s) for life, s in zip(lives, starts)]
    tick.run_many([(life.body.stepper, day) for life, day in zip(lives, days)],
                  waits=[(lambda life=life: life.body.hold_routine(1.0)) for life in lives])
  else:
    first.run(start=starts[0], max_sim_time=sim_s)
  for w in watches:
    w.finish()
  return {"seed": seed, "pair": pair, "simS": round(float(data.time), 1),
          "wallS": round(time.monotonic() - wall0, 2),
          "calls": [len(m.calls_made) for _, m, _ in robots],
          "breaks": [sum(1 for ln in life.log if "a message came in" in ln) for life in lives],
          "episodes": [len(w.episodes) for w in watches],
          "deaths": [len(life.deaths) for life in lives]}
