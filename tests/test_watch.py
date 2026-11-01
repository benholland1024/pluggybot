"""A tag watched while the robot moves (issue #554, stage 2): `watch(tag)`
keeps the head's colour camera reading one tag while the procedure goes on,
its readings `watch.*`, where a world's facts name the verb (the
practice's). Each rule pinned by a direct call or the stub body
(docs/Testing.md), and the two a flight alone can show on the humanoid:

  1. a picture each `WATCH_HZ`th of a second and its reading a latency
     later, never oftener nor sooner, a picture that does not show the tag
     saying so and the last place kept (`perception.watch.TagWatch`);
  2. the readings are the newest reading in the stance frame, and say why
     where there is none;
  3. the verb waits for its first reading, ends with the procedure, and is
     the language's only where the facts name it -- told in words of the
     language alone;
  4. a stop condition may read it, and one that never had a reading says
     why;
  5. on the humanoid: its pictures are placed through the robot's own model
     of itself, never the simulator's truth, and draw nothing on the
     flight's noise.

The scene is `scripts/hands_check.py --view watch`.
"""

import re

import numpy as np
import pytest

from pluggybot.body import StubBody
from pluggybot.mind import practice as mp
from pluggybot.perception import watch as pw
from pluggybot.procedure import axes, lang
from pluggybot.procedure import steps as st

from test_guard import Moving
from test_hands import _bare, _fly, _Life


# ---- 1. the watch: a picture at its rate, its reading a latency later ----------------


class _Looks:
  """What a watch's pictures show: `places[n]` at its n-th picture (None:
  not shown), each call noted."""

  def __init__(self, places=None):
    self.places, self.calls = places or {}, 0

  def __call__(self, tag):
    got = self.places.get(self.calls, {"x": 1.0, "y": 0.0, "z": 0.9, "range": 0.5,
                                       "normal": (1.0, 0.0, 0.0), "edge": (0.0, 1.0, 0.0)})
    self.calls += 1
    return got


def _stepped(w, n, dt=0.002, t0=0.0, sees=True):
  out = []
  for i in range(n):
    w.step(t0 + (i + 1) * dt, sees=sees)
    out.append(w.newest)
  return out


def test_a_watch_takes_a_picture_at_its_rate_and_reads_it_a_latency_later():
  """At 0.002 s a step, a picture each 34 steps (1/15 s, rounded UP: never
  oftener than the camera) and its reading 50 steps on (0.1 s): nothing
  before, and each reading the picture's, timed at the picture."""
  look = _Looks()
  w = pw.TagWatch(0.002, look)
  assert (w.every, w.lag) == (34, 50)
  w.start(53)
  got = _stepped(w, 120)
  assert look.calls == 4, "pictures at steps 0, 34, 68 and 102"
  assert got[49] is None and got[50] is not None, "the first reading 0.1 s after its picture"
  assert got[50]["at"] == pytest.approx(0.002) and got[50]["seen"] and got[50]["tag"] == 53
  assert got[84]["at"] == pytest.approx(0.002 * 35), "the next a picture's period on"
  assert got[83]["at"] == pytest.approx(0.002)
  # a period that does not divide the step rounds UP too
  w = pw.TagWatch(0.003, _Looks())
  assert w.every * 0.003 >= 1 / pw.WATCH_HZ and (w.every - 1) * 0.003 < 1 / pw.WATCH_HZ
  assert w.lag * 0.003 >= pw.LATENCY_S and (w.lag - 1) * 0.003 < pw.LATENCY_S


def test_a_picture_without_the_tag_says_so_and_the_last_place_is_kept():
  """Out of view or hidden, `seen` is False, and where it was last seen is
  kept with its own time; seen again, the new place."""
  look = _Looks({1: None, 2: None, 3: {"x": 2.0, "y": 0.0, "z": 1.0, "range": 0.6,
                                       "normal": (1.0, 0.0, 0.0), "edge": (0.0, 1.0, 0.0)}})
  w = pw.TagWatch(0.002, look)
  w.start(53)
  got = _stepped(w, 160)
  first, hidden, again = got[50], got[84], got[152]
  assert first["seen"] and first["last"]["x"] == 1.0
  assert not hidden["seen"] and hidden["last"] == first["last"], "the last place, kept"
  assert hidden["at"] == pytest.approx(0.002 * 35) and hidden["last"]["at"] == pytest.approx(0.002)
  assert again["seen"] and again["last"]["x"] == 2.0 and again["last"]["at"] == again["at"]


def test_a_get_up_sees_nothing_and_another_tag_or_a_stop_forgets_the_last():
  """While the get-up policy has the body no picture is decoded (`sees`);
  a new tag, or a stop, begins from nothing."""
  look = _Looks()
  w = pw.TagWatch(0.002, look)
  w.start(53)
  got = _stepped(w, 51, sees=False)
  assert look.calls == 0 and not got[50]["seen"] and got[50]["last"] is None
  w.start(54)
  assert w.newest is None and w.tag == 54
  got = _stepped(w, 120)
  assert next(g for g in got if g is not None)["tag"] == 54
  w.stop()
  assert w.tag is None and w.newest is None and _stepped(w, 60)[-1] is None


def test_the_cameras_frames_run_on_whatever_is_watched():
  """A watch begun takes its first picture at the camera's next frame:
  begun just after another, never sooner than a frame after the last
  picture -- restarted at each `start`, one took a picture 21 steps after
  the last (#554's review) -- and frames run on while nothing is watched."""
  look = _Looks()
  w = pw.TagWatch(0.002, look)
  w.start(53)
  _stepped(w, 55)
  assert look.calls == 2, "pictures at steps 0 and 34"
  w.start(54)
  got = _stepped(w, 13)
  assert look.calls == 2 and all(g is None for g in got), "no picture before step 68"
  _stepped(w, 1)
  assert look.calls == 3
  w.stop()
  _stepped(w, 40)
  w.start(55)
  got = _stepped(w, 300)
  first = next(i for i, g in enumerate(got) if g is not None)
  # stopped at step 69, the clock ran on: begun at step 109, its first
  # picture at 136, read 50 steps on
  assert first == 136 + w.lag - 109


# ---- 2. the readings ------------------------------------------------------------------


def _sighting(seen=True, at=0.5, last_at=0.4, x=0.6, y=0.1, z=0.9):
  return {"tag": 53, "seen": seen, "at": at,
          "last": {"x": x, "y": y, "z": z, "range": 0.55, "normal": (1.0, 0.0, 0.0),
                   "edge": (0.0, 1.0, 0.0), "at": last_at}}


def test_the_readings_are_the_newest_reading_in_the_stance_frame():
  """Its place, look and edges where the newest picture that showed it put
  them, turned into the stance frame now; `seen` the newest picture's
  word, and `age` how long ago it was last seen."""
  body = StubBody(*_bare(), pose=(1.0, 2.0, np.pi / 2))
  life = _Life(body)
  body.data.time = 0.7
  body.sight = _sighting(seen=False, x=1.0, y=2.6, z=0.9)
  read = {n: axes.SENSORS[n].read(life) for n in axes.WATCH_SENSORS}
  assert (read["watch.x"], read["watch.y"], read["watch.z"]) == pytest.approx((0.6, 0.0, 0.9))
  assert (read["watch.nx"], read["watch.ny"], read["watch.nz"]) == pytest.approx((0.0, -1.0, 0.0))
  assert (read["watch.ex"], read["watch.ey"]) == pytest.approx((1.0, 0.0))
  assert read["watch.range"] == 0.55 and read["watch.seen"] == 0.0
  assert read["watch.age"] == pytest.approx(0.3)


def test_a_reading_with_none_says_why():
  body = StubBody(*_bare())
  life = _Life(body)
  with pytest.raises(axes.NoReading, match=r"watch\.seen has no reading: `watch` has not run"):
    axes.SENSORS["watch.seen"].read(life)
  body.sight = {"tag": 53, "seen": False, "at": 0.1, "last": None}
  assert axes.SENSORS["watch.seen"].read(life) == 0.0
  with pytest.raises(axes.NoReading, match="watch.x has no reading: tag 53 has not been seen "
                                           "since `watch` began"):
    axes.SENSORS["watch.x"].read(life)


# ---- 3. the verb ----------------------------------------------------------------------


class _Watching(StubBody):
  """A stub whose watch's first reading comes `after` s in: `reading`."""

  def __init__(self, reading, after=0.1):
    super().__init__(*_bare())
    self.reading, self.after, self.began = reading, after, None

  def watch(self, tag):
    super().watch(tag)
    self.began = None if tag is None else float(self.data.time)

  def sighting(self):
    if self.began is None or self.data.time - self.began < self.after - 1e-9:
      return None
    return self.reading


def test_watch_waits_for_its_first_reading_and_ends_with_the_procedure():
  """It waits for its first reading -- ok where that shows the tag, its
  place in the stance frame -- and the procedure's end stops the watch."""
  body = _Watching(_sighting())
  ran = _fly('def a():\n    watch(53)\n    s = read("watch.seen")\n', body)
  assert ran["ok"] and ran["locals"] == {"s": 1.0}
  step = ran["steps"][0]
  assert step["ok"] and step["tag"] == 53 and step["at"] == [0.6, 0.1, 0.9]
  assert body.data.time == pytest.approx(0.1, abs=st.WATCH_POLL_S + 1e-9)
  assert body.watched == [53, None], "stopped at the procedure's end"


def test_watch_out_of_view_fails_and_watches_on_and_waits_no_longer_than_it_may():
  """Not in view as it begins, it fails -- an `if`'s test watching on, a
  failed step ending the procedure and its watch with it -- and waits no
  longer than `WATCH_FIRST_S` for a first reading."""
  body = _Watching({"tag": 53, "seen": False, "at": 0.0, "last": None})
  ran = _fly('def a():\n    if not watch(53):\n        s = read("watch.seen")\n', body)
  assert ran["ok"] and ran["locals"] == {"s": 0.0}
  assert ran["steps"][0]["reason"] == "tag 53 is not in view of your head camera now"
  body = _Watching({"tag": 53, "seen": False, "at": 0.0, "last": None})
  ran = _fly('def a():\n    watch(53)\n    s = read("watch.seen")\n', body)
  assert not ran["ok"] and ran["failedAt"] == 0 and body.watched == [53, None]
  body = _Watching(_sighting(), after=9.0)
  ran = _fly('def a():\n    watch(53)\n', body)
  assert not ran["ok"] and body.data.time == pytest.approx(st.WATCH_FIRST_S, abs=1e-6)
  assert _fly('def a():\n    watch(53.5)\n', body)["steps"][0]["reason"] == \
    "a tag is a whole number, not 53.5"


def test_watch_is_the_languages_only_where_the_facts_name_it():
  """Behind the practice's switch (`hand_facts(watching=)`), as `measure`
  is: unwatched, no verb and no readings, and the served robots' verbs and
  sensors are as they were."""
  src = 'def a():\n    watch(53)\n    s = read("watch.seen")\n'
  lang.compile_procedure(src, st.hand_facts())
  with pytest.raises(st.Refused) as e:
    lang.compile_procedure(src, st.hand_facts(watching=False))
  assert "unknown verb 'watch'" in str(e.value) and "unknown sensor 'watch.seen'" in str(e.value)
  assert "watch" not in st.VERBS
  facts = st.hand_facts()
  assert facts.verbs[facts.verbs.index("spot") + 1] == "watch"
  assert set(axes.WATCH_SENSORS) <= set(facts.sensors)


def test_the_robot_is_told_the_watch_in_words_of_the_language_alone():
  """Its rate and its latency off their constants, its stop conditions'
  rate where the move may stop on one, and not a word of a job; unwatched,
  nothing of it."""
  told = mp.system_prompt(st.hand_facts())
  bare = mp.system_prompt(st.hand_facts(watching=False))
  assert "watch" not in bare
  doc = st.HANDS_VERBS["watch"].doc
  assert f"{pw.WATCH_HZ:g} times a second" in doc and f"{pw.LATENCY_S:g} s after" in doc
  assert f"\n  watch(tag)  -- {doc}" in told
  assert mp.WATCHED % {"hz": pw.WATCH_HZ} in " ".join(told.split())
  texts = [doc] + [axes.SENSORS[n].doc for n in axes.WATCH_SENSORS]
  for text in texts:
    for word in ("lid", "chest", "knob", "hinge", "drawer", "door", "lever", "angle"):
      assert not re.search(rf"\b{word}\b", text.lower()), (word, text)
  index = mp.vocabulary_index(st.hand_facts())
  assert [v["verb"] for v in index["verbs"]].count("watch") == 1
  assert {s["name"] for s in index["sensors"]} >= set(axes.WATCH_SENSORS)


# ---- 4. a stop condition reads it -----------------------------------------------------


class _MovingWatch(Moving):
  """`Moving` (test_guard's) with a watch's newest reading: the tag's
  height `z(s)`, `s` seconds into the watch, while seen."""

  def __init__(self, z):
    super().__init__(*_bare())
    self.z, self.since = z, 0.0

  def watch(self, tag):
    super().watch(tag)
    self.since = float(self.data.time)

  def sighting(self):
    if not self.watched or self.watched[-1] is None:
      return None
    t = float(self.data.time)
    return {"tag": 53, "seen": True, "at": t,
            "last": {"x": 0.5, "y": 0.0, "z": self.z(t - self.since), "range": 0.5,
                     "normal": (1.0, 0.0, 0.0), "edge": (0.0, 1.0, 0.0), "at": t}}


def test_a_move_may_stop_on_a_watch_reading_read_at_its_cameras_rate():
  """`until="watch.z"` is a stop condition like any reading that changes
  as the robot moves -- read once each picture's period, never oftener."""
  every = int(np.ceil(1 / pw.WATCH_HZ / 0.002 - 1e-6))
  body = _MovingWatch(lambda s: 0.5 + s)
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n    watch(53)\n'
             '    move_hand("left", 0.3, 0.2, 2.0, until="watch.z", above=1.0)\n', body)
  move = ran["steps"][-1]
  assert move["ok"] and move["why"] == "until" and move["until"].startswith("watch.z read 1.")
  # read from the move's start at a picture's period: the first past 1.0 is
  # the first such reading 0.5 s in -- never a hand's every 10 ms
  first = every * (int(0.5 / (every * 0.002)) + 1)
  assert float(move["until"].split()[2]) == pytest.approx(0.5 + first * 0.002, abs=1e-3)


def test_a_stop_condition_that_never_had_a_reading_says_why():
  """A watched tag not yet seen is no reading past the number: the move
  ends as it would have, and its verdict says why the condition never read."""
  body = Moving(*_bare())
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9, until="watch.z", above=1.0)\n', body)
  reach = ran["steps"][0]
  assert reach["ok"] and "why" not in reach
  assert reach["until"] == "watch.z above 1 never had a reading: `watch` has not run"
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n'
             '    pull("left", 0, 0, 1, 10, 0.05, until="left.z", above=9)\n', Moving(*_bare()))
  assert "until" not in ran["steps"][-1], "a condition that read and never came true says nothing"


class _FallsAfterStop(Moving):
  """`Moving`, whose carry, stopped by its reading, then ends as a fall
  in the same breath -- the step after the stop."""

  def move_hand_routine(self, side, pos, stiffness=1.0, speed=0.06, patience=20.0,
                        path=None, turns=None):
    rep = yield from super().move_hand_routine(side, pos, stiffness, speed, patience, path, turns)
    return {**rep, "why": "fell"} if rep["why"] == "until" else rep


def test_a_move_another_end_took_says_nothing_of_its_stop_condition():
  """A move stopped by its reading that a fall then ended is the fall's:
  `move_hand.stopped` 0, and no word of what its condition read (#554's
  review: the verdict said both)."""
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n'
             '    move_hand("left", 0.3, 0.2, 2.0, until="left.z", above=1.0)\n',
             _FallsAfterStop(*_bare()))
  move = ran["steps"][-1]
  assert move["why"] == "fell" and not move["ok"] and "until" not in move, move


# ---- 5. on the humanoid ---------------------------------------------------------------


def _counter():
  from pluggybot.evaluation import counter as ct
  from pluggybot.evaluation.practice import ProcedureLife
  model, data, body = ct.world(ct.set_out(0))
  body.run(body.hold_routine(0.5))
  return model, data, body, ProcedureLife(body)


def test_a_watch_reads_the_robots_senses_never_the_simulators_truth(monkeypatch):
  """Hardware-honest (#554, Ben): each picture placed through the robot's
  own model of itself and its stance, and every reading a procedure takes
  and every stop it asks for, with the body's every view of the world's
  `MjData` a spy -- nothing of where the world says the bodies or the
  camera are (`test_humanoid.TRUTH`, and the camera's pose). The picture is
  the camera's: its render reads the world, as a camera sees it."""
  from pluggybot.perception.tags import TagDetector
  from test_humanoid import TRUTH, _Spy
  model, data, body, life = _counter()
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
  frame = TagDetector.frame
  monkeypatch.setattr(TagDetector, "frame", lambda self, d: frame(self, getattr(d, "_data", d)))
  body.mission.sight._look = spied(body.mission.sight._look)
  for name in axes.WATCH_SENSORS:
    monkeypatch.setitem(axes.SENSORS, name, axes.Sensor(
      name, spied(axes.SENSORS[name].read), axes.SENSORS[name].doc,
      every_s=axes.SENSORS[name].every_s))
  facts = st.hand_facts()
  proc = lang.compile_procedure(
    'def a():\n    watch(53)\n    wait(0.2)\n'
    '    s = read("watch.seen")\n    x = read("watch.x")\n    n = read("watch.nx")\n'
    '    r = read("watch.range")\n    g = read("watch.age")\n'
    '    reach("right", 0.30, -0.25, 1.00, until="watch.range", above=0.01)\n', facts)
  ran = body.run(lang.run_procedure_routine(life, proc, facts))
  assert ran["ok"] and ran["locals"]["s"] == 1.0, ran
  assert 0.1 <= ran["locals"]["g"] <= 0.1 + 2 / pw.WATCH_HZ
  assert 0.2 < ran["locals"]["x"] < 0.7 and 0.3 < ran["locals"]["r"] < 1.0
  assert ran["steps"][-1]["why"] == "until"
  assert ran["steps"][-1]["until"].startswith("watch.range read 0.")
  assert spy.read, "nothing was read through the spy"
  assert not spy.read & (TRUTH | {"cam_xpos", "cam_xmat"}), spy.read


def test_a_picture_before_the_arms_loop_has_posed_the_robots_model_shows_nothing():
  """A stand-up builds a fresh reach, whose model of itself its loop has
  not yet posed off the senses: a picture then cannot be placed, and shows
  nothing -- never one placed through a model at its default pose."""
  import mujoco
  from pluggybot.body import body_for
  from pluggybot.humanoid import model as hm
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  body = body_for(model, data)
  asked = []
  body._decoded = lambda tag, at_height=None, fresh=True: asked.append(fresh) or {"x": 0.0}
  assert all(g is None for g in body.reach.sensed_grip.values())
  assert body._sighted(53) is None and asked == []
  body.run(body.hold_routine(0.02))
  assert body._sighted(53) == {"x": 0.0} and asked == [False]


def test_a_watchs_pictures_draw_nothing_on_the_flights_noise():
  """A watch whose pictures show the tag leaves the flight as one whose
  pictures show nothing, bit for bit: each is placed through the robot's
  model of itself as its arm's loop last posed it (#554: posed afresh, a
  picture drew on its encoders' shared noise and moved the flight)."""
  src = ('def a():\n    if watch(53):\n        pass\n'
         '    reach("right", 0.30, -0.25, 1.00)\n    withdraw("right")\n')
  facts = st.hand_facts()
  ends = []
  for blind in (False, True):
    model, data, body, life = _counter()
    if blind:
      body.mission.sight._look = lambda tag: None
    ran = body.run(lang.run_procedure_routine(life, lang.compile_procedure(src, facts), facts))
    assert ran["steps"][0]["ok"] is (not blind)
    ends.append(np.concatenate([data.qpos, data.qvel, data.ctrl]).tobytes())
    body.close()
  assert ends[0] == ends[1]
