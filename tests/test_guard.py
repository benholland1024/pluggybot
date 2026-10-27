"""A hand's move that stops on a reading (issue #554, stage 1): `reach`,
`move_hand` and `pull` may each take one stop condition -- a sensor `read`
names, and the number it stops past, above or below -- where a world's facts
say so (`WorldFacts.guarded`, the practice's). Each rule pinned by a direct
call, the stub body or a `Reach` with no world (docs/Testing.md):

  1. the language takes it only where the facts allow it, as one sensor
     that changes while the hand moves and one side of one number, which
     may be computed -- and refuses anything else with every reason;
  2. the sensor is read as often as its part reads it and no oftener, from
     the move's start, and the first reading past its number stops the
     hand -- once a move is under way to stop;
  3. a move stopped so is ok and says so, in its verdict and in
     `<verb>.stopped`; one whose reading never came ends as it would have;
  4. the hand is held where it is: a carry's and a pull's where the hand
     IS, a reach's where its arm is commanded, out either way;
  5. the robot is told it in words of the language alone, behind the flag.

The flights that measured it are `scripts/practice.py`'s and the scene is
`scripts/hands_check.py --view until`.
"""

import math
import re
from types import SimpleNamespace as NS

import numpy as np
import pytest

from pluggybot.body import StubBody
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import reach as rc
from pluggybot.mind import practice as mp
from pluggybot.procedure import axes, lang
from pluggybot.procedure import steps as st

from test_hands import _bare, _bare_reach, _fly, _out_hand, _standing_reach


# ---- 1. the language -----------------------------------------------------------------


def test_a_move_takes_a_stop_condition_only_where_the_facts_allow_one():
  """Behind the practice's flag (`WorldFacts.guarded`), as the verb tests
  and the maths are: unguarded, `until` is an argument the verb does not
  take, its readings are no sensors, and the served robots' verbs are as
  they were."""
  src = ('def a():\n    reach("left", 0.3, 0.2, 0.9, until="left.fx", above=8)\n'
         '    s = read("reach.stopped")\n')
  lang.compile_procedure(src, st.hand_facts())
  with pytest.raises(st.Refused) as e:
    lang.compile_procedure(src, st.hand_facts(guarded=False))
  assert e.value.reasons[:2] == ["line 2, reach: takes no until", "line 2, reach: takes no above"]
  assert "unknown sensor 'reach.stopped'" in str(e.value)
  assert not set(axes.STOPPED_SENSORS) & set(st.hand_facts(guarded=False).sensors)
  assert set(axes.STOPPED_SENSORS) <= set(st.hand_facts().sensors)
  for verb in st.VERBS.values():
    assert not set(verb.args) & set(st.GUARD_ARGS), verb.name
  for name, verb in st.GUARDED_VERBS.items():
    assert verb.run is st.HANDS_VERBS[name].run
    assert list(verb.args) == list(st.HANDS_VERBS[name].args) + ["until", "above", "below"]


@pytest.mark.parametrize("call, reasons", [
  ('pull("left", 0, 0, 1, 10, 0.2, until="left.z")',
   ["until names the sensor a move stops on: give the number it stops past as above=N or "
    "below=N"]),
  ('pull("left", 0, 0, 1, 10, 0.2, below=1)',
   ['above and below are the number a move stops past: name the sensor it reads with '
    'until="<sensor>"']),
  ('pull("left", 0, 0, 1, 10, 0.2, until="left.z", above=1, below=0.5)',
   ["a move stops past its number on one side: above or below, not both"]),
  ('move_hand("left", 0.3, 0.2, 1.0, until="pull.force", above=5)',
   ["until='pull.force' is what a verb last reported, which no move changes"]),
  ('reach("left", 0.3, 0.2, 1.0, until="left.temp", above=5)',
   ["until='left.temp' is no sensor you have (a move may stop on time, left.x"]),
  ('reach("left", 0.3, 0.2, 1.0, until=n, above=5)', ["until takes a name in quotes"]),
])
def test_a_stop_condition_is_one_sensor_that_moves_and_one_side_of_one_number(call, reasons):
  """Every reason at once, before anything moves, each naming its call: a
  sensor and a number, never one without the other or both sides; and a
  sensor a move can change -- what a verb last reported never does, so a
  move would stop at once or never."""
  with pytest.raises(st.Refused) as e:
    lang.compile_procedure(f"def a():\n    n = 1\n    {call}\n", st.hand_facts())
  for want, got in zip(reasons, e.value.reasons):
    assert got.startswith(f"line 3, {call.split('(')[0]}: {want}"), got
  assert len(e.value.reasons) == len(reasons)


def test_the_number_may_be_computed_and_is_checked_when_it_is():
  """The robot turns what it wants into a number itself (Ben, 2026-10-27):
  `above` may be any expression, and is checked as every computed argument
  is, when the call runs."""
  facts = st.hand_facts()
  lang.compile_procedure('def a():\n    z = read("left.z")\n'
                         '    move_hand("left", 0.3, 0.2, 1.2, until="left.z", above=z + 0.1)\n',
                         facts)
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n    z = 0\n'
             '    move_hand("left", 0.3, 0.2, 1.2, until="left.z", above=1 / z)\n')
  assert ran["steps"][-1] == {"i": 1, "verb": "move_hand", "line": 4, "ok": False,
                              "reason": "division by zero"}


def test_a_sensors_rate_is_its_parts_and_a_hands_is_once_its_arms_command():
  """A reading a move may stop on carries how often its part reads it; a
  verb's report carries none. A hand's place, force and grip are fresh once
  a command of its arm's, the reach's own period."""
  assert axes.HAND_EVERY_S == rc.EVERY_S
  live = [s for s in st.hand_facts().sensors if axes.SENSORS[s].every_s is not None]
  assert live == ["time"] + [f"{h}.{k}" for h in hm.SIDES
                             for k in ("x", "y", "z", "fx", "fy", "fz", "held", "gap")]
  assert {axes.SENSORS[s].every_s for s in live} == {rc.EVERY_S}
  for s in st.hand_facts().sensors:
    if s.split(".")[0] in ("spot", "pull", "step", "object", "reach", "move_hand"):
      assert axes.SENSORS[s].every_s is None, s


# ---- 2. the guard: read at its rate, the first reading past stops the hand ----------


class _Guarded:
  """What a guard reads and stops: a clock, a sensor a test sets, and a body
  whose `stop_hand` says whether a move was under way."""

  def __init__(self, monkeypatch, under_way=(True,), every_s=0.01):
    self.value, self.reads, self.stops = 0.0, [], []
    self.model, self.data = NS(opt=NS(timestep=0.002)), NS(time=0.0)
    under = iter(under_way)
    self.body = NS(stop_hand=lambda side: self.stops.append((self.data.time, side))
                   or next(under, True))
    monkeypatch.setitem(axes.SENSORS, "test.reading", axes.Sensor(
      "test.reading", lambda life: self.reads.append(round(life.data.time, 6)) or self.value,
      "a test's", every_s=every_s))

  def fly(self, routine, side="above", number=1.0, values=None):
    """`routine` under a guard, a physics step a yield, the reading
    `values[t]` at step t."""
    gen = st._guarded_routine(self, "left", ("test.reading", side, number), routine)
    sent, n = None, 0
    while True:
      self.value = (values or {}).get(n, self.value)
      try:
        gen.send(sent)
      except StopIteration as stop:
        return stop.value
      self.data.time += self.model.opt.timestep
      n += 1


def _steps(n, out="done"):
  for _ in range(n):
    yield ()
  return out


def test_a_stop_condition_is_read_at_its_sensors_rate_from_the_moves_start(monkeypatch):
  """Once the move has begun and then every `every_s` -- five physics
  steps for a hand's 10 ms -- never oftener, and the first reading past its
  number, strictly, stops the hand: the stop is asked once, and the verb
  ends when its routine does."""
  g = _Guarded(monkeypatch)
  got, stopped = g.fly(_steps(40), values={0: 0.5, 13: 1.0, 16: 1.2})
  assert got == "done"
  assert g.reads == pytest.approx([0.0, 0.01, 0.02, 0.03, 0.04], abs=1e-9), \
    "read at its rate, and never again once it stopped"
  assert g.stops == [(pytest.approx(0.04), "left")]
  assert stopped == {"read": 1.2, "s": pytest.approx(0.04)}
  # below: past is less than its number; equal is not past
  g = _Guarded(monkeypatch)
  _, stopped = g.fly(_steps(40), side="below", number=0.5, values={0: 0.5, 7: 0.49})
  assert stopped["read"] == 0.49 and stopped["s"] == pytest.approx(0.02)
  # a reading that never comes ends the move as it would have ended
  g = _Guarded(monkeypatch)
  assert g.fly(_steps(12), values={0: 0.0}) == ("done", None) and g.stops == []


def test_a_stop_condition_already_met_stops_the_move_once_one_is_under_way(monkeypatch):
  """Read at once: past its number already, the hand does not move -- but
  only a move under way is stopped (`stop_hand` says), and until one is, it
  is read again at its rate: a reach's first steps stand the body up."""
  g = _Guarded(monkeypatch, under_way=(False, False, True))
  _, stopped = g.fly(_steps(40), values={0: 2.0})
  assert [t for t, _ in g.stops] == pytest.approx([0.0, 0.01, 0.02])
  assert stopped["s"] == pytest.approx(0.02)


def test_a_slower_part_is_read_at_its_own_rate(monkeypatch):
  """A reading a part refreshes slower -- a camera's frames (stage 2) -- is
  read no oftener than it is fresh: 33 ms is 17 physics steps, never 16."""
  g = _Guarded(monkeypatch, every_s=0.033)
  g.fly(_steps(60), values={0: 0.0})
  assert g.reads == pytest.approx([0.0, 0.034, 0.068, 0.102], abs=1e-9)


def test_a_guard_drives_its_routine_as_yield_from_would(monkeypatch):
  """Sent, thrown into and closed as `yield from` would (`_felt_routine`'s
  rule): a hook's `MissionAborted` reaches the body's routine, whose
  `finally` runs; and a move with no stop condition is the routine alone."""
  ended = []

  def routine():
    try:
      yield "a"
      try:
        yield "b"
      except KeyError:
        yield "caught"
      return "done"
    finally:
      ended.append(True)
  g = _Guarded(monkeypatch)
  for guard in (("test.reading", "above", 9.0), None):
    gen = st._guarded_routine(g, "left", guard, routine())
    assert next(gen) == "a" and gen.send(None) == "b"
    assert gen.throw(KeyError("x")) == "caught"
    with pytest.raises(StopIteration) as stop:
      gen.send(None)
    assert stop.value.value == ("done", None)
    inner = routine()
    gen = st._guarded_routine(g, "left", guard, inner)
    next(gen)
    gen.close()
  assert ended == [True] * 4


# ---- 3. the verbs: ok, and said ---------------------------------------------------------


class Moving(StubBody):
  """A stub whose hands move: a reach from where the hand hangs, a carry
  and a pull, each at `SPEED` a physics step at a time until it gets there
  -- or `stop_hand` stops it where it is."""

  SPEED = 0.2

  def __init__(self, *a, **kw):
    super().__init__(*a, **kw)
    self.moving: dict[str, bool] = {}

  def stop_hand(self, side) -> bool:
    if not self.moving.get(side):
      return False
    self.moving[side] = False
    return True

  def _go(self, side, to):
    """Walked toward `to`; whether it got there."""
    to = np.asarray(to, dtype=float)
    self.moving[side] = True
    dt = self._model.opt.timestep
    while self.moving[side]:
      at = np.asarray(self.hands[side], dtype=float)
      gap = float(np.linalg.norm(to - at))
      if gap <= self.SPEED * dt:
        self.hands[side] = tuple(to)
        self.moving[side] = False
        return True
      self.hands[side] = tuple(at + (to - at) * (self.SPEED * dt / gap))
      yield self.STILL
    return False

  def reach_routine(self, side, pos, rot=None, stiffness=1.0, patience=20.0, path=None,
                    turns=None):
    self.hands[side] = self.hand_at(side)
    there = yield from self._go(side, pos)
    self._why[side] = "" if there else "until"
    return there

  def move_hand_routine(self, side, pos, stiffness=1.0, speed=0.06, patience=20.0,
                        path=None, turns=None):
    if side not in self.hands:
      return {"why": "not out", "seconds": 0.0, "moved": 0.0, "dx": 0.0, "dy": 0.0, "dz": 0.0}
    a = np.asarray(self.hand_at(side), dtype=float)
    there = yield from self._go(side, pos)
    d = np.asarray(self.hand_at(side)) - a
    return {"why": "arrived" if there else "until", "seconds": 0.0,
            "moved": float(np.linalg.norm(d)), "dx": d[0], "dy": d[1], "dz": d[2]}

  def pull_routine(self, side, direction, limit_n, distance_m, speed=0.06, patience=30.0,
                   give=True):
    a = np.asarray(self.hand_at(side), dtype=float)
    w = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    there = yield from self._go(side, a + distance_m * w)
    d = np.asarray(self.hand_at(side)) - a
    return {"why": "distance" if there else "until", "seconds": 0.0,
            "moved": float(np.linalg.norm(d)), "along": float(d @ w), "dx": d[0],
            "dy": d[1], "dz": d[2], "direction": list(w), "force": 2.0}


def test_a_move_stopped_on_its_reading_is_ok_and_says_what_it_read():
  """Each of the three, stopped where its reading crossed -- within the
  step its sensor was last read at -- is ok, says so in its verdict and in
  `<verb>.stopped`, and holds there; a move whose reading never came ends
  as it would have, `<verb>.stopped` 0."""
  ran = _fly('def a():\n'
             '    reach("left", 0.3, 0.2, 1.3, until="left.z", above=1.0)\n'
             '    r = read("reach.stopped")\n'
             '    z1 = read("left.z")\n'
             '    move_hand("left", 0.3, 0.6, 1.0, until="left.y", above=0.4)\n'
             '    m = read("move_hand.stopped")\n'
             '    y = read("left.y")\n'
             '    pull("left", 0, -1, 0, 10, 0.5, until="left.y", below=0.3)\n'
             '    p = read("pull.stopped")\n'
             '    y2 = read("left.y")\n'
             '    pull("left", 0, 0, 1, 10, 0.05, until="left.z", above=9)\n'
             '    q = read("pull.stopped")\n', Moving(*_bare()))
  assert ran["ok"]
  reach, move, pull, again = (s for s in ran["steps"] if s["verb"] != "expr")
  got = ran["locals"]
  step = Moving.SPEED * 0.002 * round(axes.HAND_EVERY_S / 0.002)
  assert (got["r"], got["m"], got["p"], got["q"]) == (1, 1, 1, 0)
  assert 1.0 < got["z1"] <= 1.0 + step
  assert 0.4 < got["y"] <= 0.4 + step and 0.3 - step <= got["y2"] < 0.3
  assert reach["ok"] and reach["why"] == "until" and re.fullmatch(
    r"left\.z read 1\.0\d* \(above 1\) \d\.\d\d s in", reach["until"]), reach["until"]
  assert move["ok"] and move["why"] == "until" and move["until"].startswith("left.y read 0.4")
  assert pull["ok"] and pull["why"] == "until" and "(below 0.3)" in pull["until"]
  assert again["ok"] and again["why"] == "distance" and "until" not in again


def test_a_condition_met_as_a_move_begins_stops_it_before_it_moves():
  """Read once the move has begun, never before: the body's routine sets its
  move up and yields its first command, and a reading already past stops it
  there -- read before, nothing was under way to stop, and the hand moved
  until the next reading."""
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n'
             '    move_hand("left", 0.3, 0.2, 1.3, until="left.z", above=0.5)\n',
             Moving(*_bare()))
  move = ran["steps"][-1]
  assert move["ok"] and move["why"] == "until"
  assert move["moved"] <= Moving.SPEED * 0.002 + 1e-9, "one command's step at most"


def test_a_move_whose_reading_never_comes_fails_as_it_would_have():
  """A stop condition takes nothing from a move's own ends: a reach out of
  reach still fails, saying why, `reach.stopped` 0."""
  body = Moving(*_bare())
  body.out_of_reach = {"left"}

  def reach_routine(side, pos, rot=None, stiffness=1.0, patience=20.0, path=None, turns=None):
    self = body
    self._why[side] = "out of reach"
    return False
    yield
  body.reach_routine = reach_routine
  ran = _fly('def a():\n    if not reach("left", 0.3, 0.2, 1.3, until="left.z", above=1.0):\n'
             '        s = read("reach.stopped")\n', body)
  (step,) = ran["steps"]
  assert not step["ok"] and step["why"] == "out of reach" and ran["locals"]["s"] == 0


def test_a_stopped_reading_is_none_before_its_verb_and_its_verbs_alone():
  """`<verb>.stopped` before its verb ran stops the run saying so, as every
  report's sensor does; and each verb's is its own."""
  ran = _fly('def a():\n    wait(0)\n    s = read("pull.stopped")\n')
  assert ran["steps"][-1]["reason"] == "pull.stopped has no reading: `pull` has not run"
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 1.3, until="left.z", above=1.0)\n'
             '    s = read("move_hand.stopped")\n', Moving(*_bare()))
  assert ran["steps"][-1]["reason"] == "move_hand.stopped has no reading: `move_hand` has not run"


def test_the_report_shows_what_stopped_a_move():
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 1.3, until="left.z", above=1.0)\n',
             Moving(*_bare()))
  text = mp.report_text(1, ran, {"ok": False, "reason": "no"}, 1.0, False, 1)
  assert re.search(r'line 2 reach: ok \{"hand": "left", "why": "until", "until": '
                   r'"left\.z read 1\.0\d* \(above 1\) [\d.]+ s in"\}', text), text


# ---- 4. the hand held where it is ---------------------------------------------------------


H = np.array([0.30, 0.20, 0.95])


def test_a_halted_carry_holds_where_the_hand_is_and_says_why():
  """As a carry's own stops end it (`Reach.stop`): where the hand IS, at the
  turn it was asked, its report's why the stop's."""
  r = _bare_reach()
  _out_hand(r, H, np.eye(3))
  there = H + [0.0, 0.0, 0.02]
  r.grip_now = lambda side: (there.copy(), np.eye(3))
  assert r.halt("left", "until")
  assert r.phase["left"] == "hold" and r.report["left"]["why"] == "until"
  assert np.allclose(r.cmd["left"].pos, there) and np.allclose(r.goal["left"].pos, there)
  assert r.out("left") and not r.halt("left", "until"), "nothing under way: nothing changed"


def test_a_halted_pull_holds_where_the_hand_is_and_reports_its_pull():
  from test_force_move import _pulling
  r = _pulling(np.array([0.0, 0.0, 1.0]))
  r.pulls["left"]["peak"] = 7.5
  assert r.halt("left", "until")
  rep = r.report["left"]
  assert rep["why"] == "until" and rep["force"] == 7.5 and "went" in rep and "way" in rep


def test_a_halted_reach_on_its_way_in_holds_at_its_command():
  """On its line in, as a hand stopped on its way is (`blocked`): its
  command where it was, its goal the reach's, out -- a carry may go on and a
  withdraw backs out along the line."""
  from pluggybot.humanoid.arms import HandGoal
  r = _bare_reach()
  r.cmd["left"] = HandGoal(H.copy(), np.eye(3), 0.5)
  r.goal["left"] = HandGoal(H + [0.08, 0.0, 0.0], np.eye(3), 0.5)
  r.phase["left"] = "approach"
  assert r.halt("left", "until")
  assert r.phase["left"] == "blocked" and r.done("left") and r.out("left")
  assert np.allclose(r.cmd["left"].pos, H) and np.allclose(r.goal["left"].pos, H + [0.08, 0, 0])


def test_a_halted_reach_on_its_way_out_holds_its_arm_where_it_is_commanded():
  """Walking out to its pre-grasp in joint space, there is no line: the arm
  held at its joints' command -- no setpoint written across a gap -- and the
  hand out at the grip they put it at, so it may be carried on or withdrawn
  as any hand that is out."""
  from pluggybot.humanoid.arms import HandGoal
  r = _standing_reach()
  r.goal["left"] = HandGoal(H.copy(), np.eye(3), 0.7)
  r.q_to["left"] = r.rest["left"] + 0.6
  r.q_cmd["left"] = r.rest["left"] + 0.25
  q = r.q_cmd["left"].copy()
  r.phase["left"] = "move"
  assert r.halt("left", "until")
  assert r.phase["left"] == "blocked" and r.out("left") and r.done("left")
  assert np.allclose(r.q_to["left"], q) and np.allclose(r.q_cmd["left"], q)
  sm = r._planner_model()
  sm._set_arm("left", q)
  pos, rot = sm.grip_pose("left")
  assert np.allclose(r.cmd["left"].pos, pos) and np.allclose(r.goal["left"].pos, pos)
  assert np.allclose(r.cmd["left"].rot, rot) and r.cmd["left"].stiffness == 0.7


@pytest.mark.parametrize("phase", ["rest", "hold", "blocked", "joints", "home", "retreat"])
def test_a_hand_not_moving_to_a_place_is_never_halted(phase):
  r = _bare_reach()
  r.phase["left"] = phase
  assert not r.halt("left", "until") and r.phase["left"] == phase


def test_the_humanoids_reach_stopped_by_its_reading_says_until():
  """Through `Body`: `stop_hand` halts the reach (`Reach.halt`), and the
  reach that ends so is not "stopped short" -- what a reach blocked on its
  way says -- but "until"."""
  import mujoco
  from pluggybot.body import body_for
  from test_humanoid_body import _Sent

  class Halting(_Sent):
    def halt(self, side, why):
      if self.phase[side] != "approach":
        return False
      self.phase[side] = "blocked"
      return True
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  body = body_for(model, data)
  body.mission.reach = Halting("approach")
  assert not body.stop_hand("left"), "nothing under way yet"
  gen = body.reach_routine("left", (0.4, 0.2, 0.9))
  next(gen)
  assert body.stop_hand("left")
  with pytest.raises(StopIteration) as stop:
    while True:
      gen.send(None)
  assert stop.value.value is False and body.hand_why("left") == "until"
  body.mission.reach = Halting("blocked")
  assert body.run(body.reach_routine("left", (0.4, 0.2, 0.9))) is False
  assert body.hand_why("left") == "stopped short", "a reach's own stop is still its own"


# ---- 5. what the robot is told -------------------------------------------------------------


def test_the_robot_is_told_the_stop_condition_behind_its_flag_and_nothing_of_a_job():
  """Generic, its example nothing a job does (#554: no lid, door or
  hinge), the rate off its constant; unguarded, not a word of it. The
  example compiles as it stands."""
  told = mp.system_prompt(st.hand_facts())
  bare = mp.system_prompt(st.hand_facts(guarded=False))
  assert "until=" not in bare and "`until`" not in bare and ".stopped" not in bare
  assert "[," not in bare
  rule = " ".join((mp.GUARD % {"rate": 100, "n": mp.GUARD_EXAMPLE_N}).split())
  assert rule in " ".join(told.split())
  assert f"{round(1 / axes.HAND_EVERY_S)} times a second" in rule
  for word in ("lid", "chest", "knob", "hinge", "arc", "handle", "box", "drawer", "door",
               "lever", "angle"):
    assert not re.search(rf"\b{word}\b", rule.lower()), word
  example = told.split("pushes on something with 10 N:\n", 1)[1].split("\n", 1)[0].strip()
  lang.compile_procedure(f"def a():\n    reach(\"right\", 0.3, -0.2, 1.2)\n    {example}\n",
                         st.hand_facts())
  for v in ("reach", "move_hand", "pull"):
    assert re.search(rf"\n  {v}\([^\n]*\[, until, above, below\]\)  -- ", told), v
    assert st.GUARD_DOC.strip(". ") in told
  for v in st.GUARDED_VERBS.values():
    for word in ("lid", "chest", "knob", "hinge", "drawer", "door", "lever", "angle"):
      assert not re.search(rf"\b{word}\b", v.doc.lower()), (v.name, word)


def test_the_example_stops_over_what_a_hand_moving_in_free_air_reads():
  """Its number is the robot's to learn, but an example that stopped a hand
  in free air would teach the wrong one: over the 4.4 N its own carry read
  that way in free air at its hardest (#554, flown)."""
  assert mp.GUARD_EXAMPLE_N >= 2 * 4.4


def test_the_index_says_which_arguments_a_verb_does_without():
  """The guide's data (rooftop-media-2026 #441): each verb's arguments in
  order, the stop condition's at its end, `optional`, with no default."""
  index = mp.vocabulary_index(st.hand_facts())
  for v in index["verbs"]:
    if v["verb"] in ("reach", "move_hand", "pull"):
      assert v["args"][-3:] == [{"name": "until", "kind": "str", "optional": True},
                                {"name": "above", "kind": "float", "optional": True},
                                {"name": "below", "kind": "float", "optional": True}]
    else:
      assert not [a for a in v["args"] if a.get("optional")], v["verb"]
  bare = mp.vocabulary_index(st.hand_facts(guarded=False))
  assert not [a for v in bare["verbs"] for a in v["args"] if a.get("optional")]
  assert math.isclose(1.0 / axes.HAND_EVERY_S, 100.0)


# ---- the one physics claim: the real hand stops where its reading crossed -----------


def test_the_humanoids_hand_stops_where_its_reading_crossed_and_holds_there():
  """Flown, the humanoid on a bare floor: a carry up and a pull up, each
  stopped on its hand's height -- read off its encoders through its model
  of itself -- end within a few millimetres of the number, the grip truly
  there (our grading, the simulator's site), and still there a second on:
  the stop holds where the hand is (#554's probe: 1.150 for 1.15, 1.1998
  for 1.2)."""
  import mujoco
  from pluggybot.body import body_for
  from pluggybot.evaluation.practice import ProcedureLife
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  body = body_for(model, data)
  facts = st.hand_facts()
  proc = lang.compile_procedure(
    'def a():\n    reach("right", 0.30, -0.25, 1.00)\n'
    '    move_hand("right", 0.30, -0.25, 1.30, until="right.z", above=1.10)\n'
    '    wait(1)\n    z1 = read("right.z")\n'
    '    pull("right", 0, 0, 1, 10, 0.4, until="right.z", above=1.20)\n'
    '    wait(1)\n    z2 = read("right.z")\n', facts)
  ran = body.run(lang.run_procedure_routine(ProcedureLife(body), proc, facts))
  _, move, _, pull, _ = ran["steps"]
  assert move["ok"] and move["why"] == "until" and pull["ok"] and pull["why"] == "until"
  assert ran["locals"]["z1"] == pytest.approx(1.10, abs=0.004)
  assert ran["locals"]["z2"] == pytest.approx(1.20, abs=0.004)
  assert float(data.site_xpos[model.site("right_grip").id][2]) == pytest.approx(1.20, abs=0.005)


def test_a_stop_condition_reads_the_robots_senses_never_the_simulators_truth(monkeypatch):
  """Hardware-honest (#554, Ben): every reading a guard takes -- the hand's
  place, its force, its grip, the clock -- and every stop it asks for, made
  with the body's every view of the world's `MjData` a spy: the drivers'
  replies, the IMU and the robot's own reckoning, nothing of where the
  world says the bodies are or what they touch (`test_humanoid.TRUTH`)."""
  import dataclasses
  import mujoco
  from pluggybot.body import body_for
  from pluggybot.evaluation.practice import ProcedureLife
  from test_humanoid import TRUTH, _Spy
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  body = body_for(model, data)
  life = ProcedureLife(body)
  spy = _Spy(data)
  holders = [body, body.mission, body.reach, body.stand, body.stand.trunk, life,
             *body.reach.grippers.values(), *body.reach.arms.values()]

  def spied(fn):
    def through_the_spy(*a, **kw):
      swapped = [(o, k) for o in holders for k, v in vars(o).items() if v is data]
      for o, k in swapped:
        setattr(o, k, spy)
      try:
        return fn(*a, **kw)
      finally:
        for o, k in swapped:
          setattr(o, k, data)
    return through_the_spy
  for name in ("right.z", "right.fz", "right.gap", "time"):
    monkeypatch.setitem(axes.SENSORS, name, dataclasses.replace(
      axes.SENSORS[name], read=spied(axes.SENSORS[name].read)))
  body.stop_hand = spied(body.stop_hand)
  facts = st.hand_facts()
  proc = lang.compile_procedure(
    'def a():\n    reach("right", 0.30, -0.25, 1.00)\n'
    '    move_hand("right", 0.30, -0.25, 1.30, until="right.z", above=1.10)\n'
    '    pull("right", 0, 0, 1, 10, 0.03, until="right.fz", above=60)\n'
    '    move_hand("right", 0.30, -0.20, 1.10, until="right.gap", below=-1)\n'
    '    move_hand("right", 0.30, -0.25, 1.10, until="time", above=0)\n', facts)
  ran = body.run(lang.run_procedure_routine(life, proc, facts))
  assert [s.get("why") for s in ran["steps"][1:]] == ["until", "distance", "arrived", "until"]
  assert spy.read, "nothing was read through the spy"
  assert not spy.read & TRUTH, spy.read & TRUTH
