"""The robot's model, graded (issue #466, stage 3): templates, the fitter, the
author and the rounds, the worker's residual, and our grading.

The flown claim -- the model over the probes' set-outs, graded with
intervals -- is the batch's (`scripts/imagine_chest.py`, SimNotes "The
robot's model, graded"). Every rule it stands on is pinned here without a
flight: the fitter against a pool whose residual is a known function of the
document's numbers, the author against a client that answers what it is
told, and the grading on worlds compiled and never stepped.
"""

import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from pluggybot.activity import chest as ch
from pluggybot.evaluation import imagined as im
from pluggybot.evaluation import model as em
from pluggybot.imagination import author as au
from pluggybot.imagination import fit as ft
from pluggybot.imagination import model as mm
from pluggybot.imagination import scene as sc
from pluggybot.imagination.record import Record
from pluggybot.imagination.rollout import Readings
from test_imagination import CABINET, hold, start  # noqa: I001 -- tests/ is on sys.path

#: A lidded box with what it cannot see left unknown.
BOX = {
  "parts": [
    {"id": "base", "shape": "box", "size": [200, 300, 140], "pos": [600, 0, 70],
     "mass": {"between": [1.0, 5.0]}},
    {"id": "lid", "shape": "slab", "size": [200, 300, 12], "pos": [600, 0, 146], "on": "base",
     "mass": {"between": [0.1, 1.0]}}],
  "joints": [
    {"id": "hinge", "type": "hinge", "part": "lid", "at": [700, 0, 140], "axis": [0, 1, 0],
     "range": [0, 100], "damping": {"between": [0.0, 0.3]},
     "friction": {"between": [0.0, 0.2]}}]}


def box(**changes):
  out = copy.deepcopy(BOX)
  out.update(changes)
  return out


# ---- templates ---------------------------------------------------------------------

def test_a_template_leaves_unknown_only_what_the_robot_cannot_see():
  t = sc.parse_template(BOX)
  assert t.names == ["base.mass", "lid.mass", "hinge.damping", "hinge.friction"]
  assert t.middle() == pytest.approx([3.0, 0.55, 0.15, 0.1])
  doc = t.fill([2.0, 0.3, 0.05, 0.04])
  scene = sc.parse(doc)
  assert scene.part("lid").mass == 0.3 and scene.joint("hinge").friction == 0.04
  # the geometry is measured, never fitted: a size, a place, an axis or a
  # range written as an unknown is refused, saying so
  for kind, i, field, value in (("parts", 1, "size", [200, 300, {"between": [8, 20]}]),
                                ("parts", 1, "pos", {"between": [0, 1]}),
                                ("joints", 0, "at", {"between": [0, 1]}),
                                ("joints", 0, "range", {"between": [0, 100]})):
    bad = box()
    bad[kind][i][field] = value
    where = "part 'lid'" if kind == "parts" else "joint 'hinge'"
    with pytest.raises(sc.Refused) as e:
      sc.parse_template(bad)
    assert any(r.startswith(f"{where}: {field} is measured, never fitted")
               for r in e.value.reasons), (field, e.value.reasons)
  bad = box()
  bad["joints"][0]["at"] = {"between": [0, 1]}
  with pytest.raises(sc.Refused) as e:
    sc.parse_template(bad)
  assert any(r.startswith("joint 'hinge': at is measured, never fitted") for r in e.value.reasons)


def test_an_unknowns_range_is_checked_at_both_ends_with_every_reason():
  bad = box()
  bad["joints"][0]["damping"] = {"between": [0.3, 0.1]}
  bad["parts"][1]["mass"] = {"between": [0.0, 1.0]}           # its low end is no mass
  bad["joints"][0]["friction"] = {"between": [0.0], "or": 1}
  with pytest.raises(sc.Refused) as e:
    sc.parse_template(bad)
  text = " | ".join(e.value.reasons)
  assert "damping between [0.3, 0.1] must be [lo, hi]" in text
  assert "a part has a mass of its own" in text
  assert "an unknown friction is" in text
  many = box(parts=[dict(BOX["parts"][0], id=f"p{i}", mass={"between": [1, 2]})
                    for i in range(sc.MAX_UNKNOWNS + 1)], joints=[])
  with pytest.raises(sc.Refused, match="at most 12"):
    sc.parse_template(many)


def test_a_document_to_compile_has_every_number():
  with pytest.raises(sc.Refused) as e:
    sc.parse(BOX)
  assert any("is unknown: a document to compile has every number" in r
             for r in e.value.reasons)


def test_a_document_drawn_in_a_frame_of_its_own_is_written_where_the_frame_puts_it():
  # The author draws in the box's frame, square to it; code puts it where
  # depth measured the box: a turn of every place, turn, point and axis.
  from pluggybot.imagination.compile import PREFIX, compile_scene
  import mujoco
  drawn = sc.parse_template(box()).fill([2.0, 0.3, 0.05, 0.04])
  drawn["parts"][1]["euler"] = [0.0, 0.0, 10.0]
  at, yaw = (1500.0, -400.0, 0.0), 30.0
  doc = sc.placed(drawn, at, yaw)
  world = compile_scene(sc.parse(doc), carrying="module_claw")
  d = mujoco.MjData(world.model)
  mujoco.mj_forward(world.model, d)
  turn = sc.rotation([0, 0, math.radians(yaw)])
  for pid in ("base", "lid"):
    p = drawn["parts"][[q["id"] for q in drawn["parts"]].index(pid)]
    want = turn @ (np.array(p["pos"]) / 1000) + np.array(at) / 1000
    assert np.allclose(d.xpos[world.model.body(f"{PREFIX}{pid}").id], want, atol=1e-9)
  lid = world.model.body(f"{PREFIX}lid").id
  assert np.allclose(d.xmat[lid].reshape(3, 3), sc.rotation([0, 0, math.radians(40.0)]),
                     atol=1e-9)
  j = world.model.joint(f"{PREFIX}hinge").id
  assert np.allclose(d.xaxis[j], turn @ [0, 1, 0], atol=1e-9)
  assert np.allclose(d.xanchor[j], turn @ np.array([0.7, 0, 0.14]) + np.array(at) / 1000,
                     atol=1e-9)
  # a template's unknowns are dynamics, and stay as they were
  assert sc.placed(BOX, at, yaw)["parts"][1]["mass"] == {"between": [0.1, 1.0]}


# ---- the fitter --------------------------------------------------------------------

class LinearPool:
  """A pool whose residual is a known straight line in the document's
  numbers: bins = M (u - truth), u each unknown's share of its range off
  the documents it is sent. `ignored` names a share it never reads."""

  def __init__(self, template, truth, ignored=(), bins=40, seed=3):
    self.t, self.truth = template, np.asarray(truth, dtype=float)
    rng = np.random.default_rng(seed)
    self.M = rng.normal(size=(2 * bins, len(truth)))
    for name in ignored:
      self.M[:, template.names.index(name)] = 0.0
    self.sent = []

  def _shares(self, doc):
    return np.array([u.share(doc[u.kind][u.index][u.field]) for u in self.t.unknowns])

  def residuals(self, docs, record, rows):
    self.sent.append(len(docs))
    return [(self.M @ (self._shares(d) - self.truth)).reshape(-1, 2) for d in docs]

  def rollouts(self, docs, record):
    return [Readings(t=np.zeros(record.n), sensed=record.sensed.copy(), joints={})
            for _ in docs]


def _flown(n=500):
  """A record of `n` rows held at a pose the arm's Jacobian is sound at."""
  sensed = np.zeros((n, 4))
  sensed[:, 2], sensed[:, 3] = 2.0, 0.5
  return Record(start=start(), dt=0.002, commands=np.zeros((n, 6)), sensed=sensed)


def test_the_fitter_finds_what_the_residual_says_and_says_what_it_cannot_see():
  t = sc.parse_template(BOX)
  truth = [0.5, 0.31, 0.72, 0.18]
  pool = LinearPool(t, truth, ignored=("base.mass",))
  got = ft.fit(t, _flown(), slice(0, 500), pool)
  for name, share in zip(t.names[1:], truth[1:]):
    u = t.unknowns[t.names.index(name)]
    assert u.share(got.values[name]) == pytest.approx(share, abs=1e-6), name
  # nothing moves the base's mass: unseen, held at its range's middle
  assert got.unseen == ("base.mass",) and got.values["base.mass"] == 3.0
  assert got.at_end == ()
  # the start was the middle and the spread, the best of them
  assert pool.sent[0] == 1 + ft.SOBOL_POINTS
  assert [s for s, _ in got.log][:2] == ["start", "robust"] and "squares" in dict(got.log)


def test_an_unknown_whose_truth_is_past_its_range_ends_at_its_end():
  t = sc.parse_template(BOX)
  got = ft.fit(t, _flown(), slice(0, 500), LinearPool(t, [0.5, 0.3, 0.6, 1.4]))
  assert got.at_end == ("hinge.friction",)
  assert got.values["hinge.friction"] == pytest.approx(0.2)


class FailingPool(LinearPool):
  """A `LinearPool` whose residual calls fail where told: call n's docs at
  the given positions come back refused."""

  def __init__(self, *a, fail=None, **kw):
    super().__init__(*a, **kw)
    self.fail = fail or {}

  def residuals(self, docs, record, rows):
    out = super().residuals(docs, record, rows)
    call = len(self.sent) - 1
    from pluggybot.imagination.rollout import Diverged
    return [Diverged("unstable") if i in self.fail.get(call, ()) else r
            for i, r in enumerate(out)]


def test_a_probe_refused_both_ways_holds_its_unknown_a_step_and_never_drops_it():
  # Its column missing, the unknown was dropped for the rest of the fit and
  # named nowhere (the review): hinge.damping stayed where the step left it.
  t = sc.parse_template(BOX)
  truth = [0.5, 0.31, 0.72, 0.18]
  # call 0 the start, call 1 the first Jacobian (an unknown a probe), call 2
  # the refused probes tried the other way
  pool = FailingPool(t, truth, fail={1: {2}, 2: {0}})
  got = ft.fit(t, _flown(), slice(0, 500), pool)
  u = t.unknowns[2]
  assert u.share(got.values[u.name]) == pytest.approx(0.72, abs=1e-6)
  assert got.held == ()
  pool = FailingPool(t, truth, fail={1: {2}, 2: {0}})
  got = ft.fit(t, _flown(), slice(0, 500), pool, max_steps=1)
  assert got.held == ("hinge.damping",), "held at the last step, and said so"


def test_the_round_kept_is_one_that_passed_every_bar_where_one_did():
  # The least over the fitted rows alone kept a round its take had failed.
  def rnd(rms, poor):
    r = mm.Round(turn=au.Turn(kind="first"))
    r.fitted = SimpleNamespace(rms=rms)
    r.judged = [{"phase": "take", "poor": poor}]
    return r
  assert mm.kept_of([rnd(0.20, True), rnd(0.25, False)]) == 1
  assert mm.kept_of([rnd(0.20, True), rnd(0.25, True)]) == 0
  assert mm.kept_of([mm.Round(turn=au.Turn(kind="first"))]) is None


def test_a_fit_is_the_same_twice():
  t = sc.parse_template(BOX)
  a = ft.fit(t, _flown(), slice(0, 500), LinearPool(t, [0.2, 0.7, 0.1, 0.9]))
  b = ft.fit(t, _flown(), slice(0, 500), LinearPool(t, [0.2, 0.7, 0.1, 0.9]))
  assert a.values == b.values and a.rollouts == b.rollouts and a.log == b.log


# ---- the worker's residual -----------------------------------------------------------

def test_a_worker_answers_a_residual_as_this_process_would_and_keeps_its_record():
  # A fit's search reads only the binned force each rollout leaves (a few
  # kB, not the rows), off a record the worker keeps -- what lets the
  # workers be a pod's, over ssh, with the same frames.
  from pluggybot.imagination.compile import compile_scene
  from pluggybot.imagination.rollout import rollout
  from pluggybot.imagination.worker import Imagination, record_id
  rec = hold(300)
  other = copy.deepcopy(CABINET)
  other["parts"][1]["mass"] = 0.9
  flown = rollout(compile_scene(sc.parse(other), carrying="module_claw"), rec)
  rec = Record(start=rec.start, dt=rec.dt, commands=rec.commands, sensed=flown.sensed)
  here = ft.residual(rec.sensed, rollout(compile_scene(sc.parse(CABINET),
                                                       carrying="module_claw"), rec),
                     slice(50, 300), rec.dt)
  with Imagination(seed=7) as w:
    there = w.residual(CABINET, rec, slice(50, 300))
    assert np.array_equal(there, here)
    assert w.held == {record_id(rec)}
    assert np.array_equal(w.residual(CABINET, rec, slice(50, 300)), here)
  with Imagination(seed=7) as w:
    # a record it is believed to hold and does not: sent again, once
    w.held = {record_id(rec)}
    assert np.array_equal(w.residual(CABINET, rec, slice(50, 300)), here)


# ---- the author ----------------------------------------------------------------------

class Client:
  """A client that answers what it is told to, in turn, and keeps every
  request it was sent."""

  def __init__(self, *answers):
    self.answers, self.sent = list(answers), []
    self.messages = SimpleNamespace(create=self.create)

  def create(self, **kw):
    self.sent.append(copy.deepcopy(kw))
    text = self.answers.pop(0)
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=100, output_tokens=50))


def answer(doc, what="a box with a lid"):
  out = {"think": "...", "what": what, "parts": [], "joints": [], "catches": []}
  for k in ("parts", "joints", "catches"):
    out[k] = [dict(item) for item in doc.get(k, [])]
  for p in out["parts"]:
    p.setdefault("euler", [0, 0, 0])
    p.setdefault("on", None)
  return json.dumps(out)


SIZES = {"origin": [1000.0, 0.0], "yaw": 0.0, "depth": 220.0, "width": 300.0, "top": 152.0,
         "tip": -50.0, "bracket_top": 150.0, "bracket_width": 12.0,
         "knob": [-60.0, 0.0, 88.0], "knob_size": 26.0}


def test_the_author_is_shown_the_picture_the_sizes_and_what_the_arm_did():
  client = Client(answer(BOX))
  a = au.Author(client, "org/model")
  turn = a.first(SIZES, "It lifted the cube.", b"\xff\xd8\xff-a-jpeg")
  assert turn.template is not None and turn.template.names[0] == "base.mass"
  sent = client.sent[0]
  content = sent["messages"][0]["content"]
  assert content[0]["type"] == "image_url" and content[1]["type"] == "text"
  assert "220.0" in content[1]["text"] and "-60.0, 0.0, 88.0" in content[1]["text"]
  assert "It lifted the cube." in content[1]["text"]
  assert sent["output_config"]["format"]["schema"] == au.answer_schema()
  # its rules prescribe nothing about what the object is
  import re
  text = au.system_prompt().lower()
  for word in ("lid", "chest", "magnet", "toy", "drop handle"):
    assert not re.search(rf"\b{word}\b", text), word


def test_the_worked_example_is_no_lidded_box_and_parses_as_a_template():
  # An example of the answer would hand it over: the example is a
  # cupboard's door, on a vertical hinge, and it is the language's own.
  raw = au.LANGUAGE.split("An example, a cupboard", 1)[1].split(":", 1)[1]
  t = sc.parse_template(au.template_of(json.loads(raw)))
  assert [j["axis"] for j in t.raw["joints"]] == [[0, 0, 1]]
  assert not t.raw["catches"]


def test_a_refused_document_goes_back_with_every_reason():
  bad = box()
  bad["joints"][0]["at"] = {"between": [0, 1]}
  bad["parts"][1]["shape"] = "sphere"
  client = Client(answer(bad), answer(BOX))
  a = au.Author(client, "org/model")
  turn = a.first(SIZES, "It lifted the cube.", b"\xff\xd8\xff-a-jpeg")
  assert [t.kind for t in a.turns] == ["first", "repair"] and turn.template is not None
  repair = client.sent[1]["messages"][-1]["content"]
  assert "never fitted" in repair and "shape must be one of" in repair
  # the picture is the first turn's alone; the conversation carries it on
  first, *later = [m for m in client.sent[1]["messages"] if m["role"] == "user"]
  assert first["content"][0]["type"] == "image_url"
  assert all(isinstance(m["content"], str) for m in later)
  client = Client(*[answer(bad)] * (au.MAX_REPAIRS + 1))
  a = au.Author(client, "org/model")
  turn = a.first(SIZES, "It lifted the cube.", None)
  assert turn.template is None and turn.refused and len(a.turns) == 1 + au.MAX_REPAIRS


def test_an_answer_that_never_came_is_asked_again_and_not_kept():
  # Shown the picture, the deployed model reasoned past 8,192 tokens and
  # answered nothing: the ask goes again, the empty answer is no turn of
  # the conversation.
  client = Client("", answer(BOX))
  a = au.Author(client, "org/model", retry_wait_s=0.0)
  turn = a.first(SIZES, "It lifted the cube.", None)
  assert [t.kind for t in a.turns] == ["first", "retry"] and turn.template is not None
  assert [m["role"] for m in client.sent[1]["messages"]] == ["user"]
  again = client.sent[1]["messages"][0]["content"]
  assert again == client.sent[0]["messages"][0]["content"] + au.RETRY_NOTE
  # retries and repairs are budgets apart: two answers that never came
  # still leave a refused document its repairs
  bad = box()
  bad["joints"][0]["at"] = {"between": [0, 1]}
  client = Client("", "", answer(bad), answer(BOX))
  a = au.Author(client, "org/model", retry_wait_s=0.0)
  turn = a.first(SIZES, "It lifted the cube.", None)
  assert [t.kind for t in a.turns] == ["first", "retry", "retry", "repair"]
  assert turn.template is not None
  assert client.sent[2]["messages"][0]["content"].count(au.RETRY_NOTE) == 1
  # ...and a first turn with its picture carries the note in its words
  client = Client("", answer(BOX))
  a = au.Author(client, "org/model", retry_wait_s=0.0)
  a.first(SIZES, "It lifted the cube.", b"\xff\xd8\xff-a-jpeg")
  parts = client.sent[1]["messages"][0]["content"]
  assert parts[0]["type"] == "image_url" and parts[1]["text"].endswith(au.RETRY_NOTE)


def test_a_revision_is_shown_the_fit_and_where_it_left_too_much():
  report = {"values": {"lid.mass": 0.4, "hinge.friction": 0.2}, "atEnd": ["hinge.friction"],
            "unseen": [], "failed": "",
            "phases": [{"label": "the first lift", "rmsN": 1.2, "barN": None, "poor": False,
                        "trace": [(0.0, (0.1, -2.4)), (0.5, (0.0, 0.05))]},
                       {"label": "the first lowering", "rmsN": 0.05, "barN": None,
                        "poor": False, "trace": [(0.0, (0.0, 0.07))]},
                       {"label": "the sweeps together", "rmsN": 0.8, "barN": 0.32, "poor": True,
                        "whole": True, "trace": [(0.0, (9.0, 9.0))]}]}
  text = au.report_text(report)
  assert "hinge.friction = 0.2  (at an end of your range)" in text
  assert "the first lift: 1.200\n" in text, "no bar where none judges"
  assert "the sweeps together: 0.800 (bar 0.320)  PAST THE BAR" in text
  # a poor whole shows the traces of the stretches it spans, never its own
  assert "0.0 s  (+0.10, -2.40)" in text and "0.0 s  (+0.00, +0.07)" in text
  assert "+9.00" not in text
  # ...and only a poor whole: where only the take was poor, its trace alone
  take = {"label": "taking hold", "rmsN": 0.08, "barN": 0.05, "poor": True,
          "trace": [(0.0, (0.5, 0.5))]}
  only_take = dict(report, phases=[take] + [dict(p, poor=False) for p in report["phases"]])
  text = au.report_text(only_take)
  assert "(+0.50, +0.50)" in text and "(+0.10, -2.40)" not in text
  # where what the real object put on the jaws is known, it is shown beside
  # the simulated one's
  report["phases"][0]["felt"] = [(0.0, (0.3, -1.2), (0.2, 1.2))]
  assert "0.0 s  real (+0.30, -1.20)   simulated (+0.20, +1.20)" in au.report_text(report)
  assert "No fit could be read" in au.report_text({**report, "failed": "it went unstable"})


# ---- the rounds --------------------------------------------------------------------

def _rounds(monkeypatch, poor, refused=()):
  """`imagine` with each round's fit stubbed: round r's rms is 1 - r/10 but
  for r in `poor`, judged poor; an answer in `refused` never parses."""
  fits = []

  def fake_fit(template, record, rows, pool, **kw):
    r = len(fits)
    fits.append(template)
    return SimpleNamespace(values={}, at_end=(), unseen=(), rms=[0.3, 0.1, 0.2][r],
                           readings=None, document={"round": r},
                           as_dict=lambda: {})
  monkeypatch.setattr(mm, "fit", fake_fit)
  monkeypatch.setattr(mm, "judge", lambda record, readings, phases, felt=None: [
    {"phase": "up0", "label": "a lift", "rmsN": 1.0, "barN": 0.5,
     "poor": len(fits) - 1 in poor, "trace": []}])
  answers = [answer(BOX) if i not in refused else answer(box(parts=[]))
             for i in range(10)]
  a = au.Author(Client(*answers), "org/model")
  m = mm.imagine(a, _flown(), [], slice(0, 500), SIZES, (1000.0, 0.0), 0.0, "did", None,
                 pool=None)
  return m, fits


def test_a_fit_left_poor_goes_back_and_the_best_round_is_kept(monkeypatch):
  m, fits = _rounds(monkeypatch, poor={0})
  assert len(m.rounds) == 2 and m.revisions == 1 and m.kept == 1
  assert m.document == {"round": 1}
  m, fits = _rounds(monkeypatch, poor={0, 1, 2})
  assert len(m.rounds) == mm.MAX_ROUNDS and m.kept == 1, "the least left, not the last"
  m, fits = _rounds(monkeypatch, poor=set())
  assert len(m.rounds) == 1 and m.revisions == 0


def test_an_author_whose_document_never_parses_ends_with_no_model(monkeypatch):
  m, fits = _rounds(monkeypatch, poor={0}, refused=set(range(10)))
  assert not fits and m.kept is None and m.document is None
  assert m.rounds[0].error.startswith("its document was refused")


def test_a_phase_past_its_bar_is_poor_and_its_trace_is_the_real_less_the_imagined():
  rec = _flown(1000)
  sensed = rec.sensed.copy()
  sensed[:, 2], sensed[:, 3] = 2.0, 0.5                    # a pose the Jacobian is sound at
  rec = Record(start=rec.start, dt=rec.dt, commands=rec.commands, sensed=sensed)
  imagined = sensed.copy()
  imagined[500:, 1] += 0.2                                 # the elbow held more, late
  phases = [mm.Phase("early", "early", (0, 500), 0.01),
            mm.Phase("late", "late", (500, 1000), None),
            mm.Phase("all", "all", (0, 1000), 0.01, whole=True)]
  got = mm.judge(rec, Readings(t=np.zeros(1000), sensed=imagined, joints={}), phases)
  assert got[0]["rmsN"] == 0.0 and not got[0]["poor"]
  assert not got[1]["poor"], "a phase with no bar is reported, never judged"
  assert got[2]["poor"]
  from pluggybot.legs.imagined import force_apart
  f = force_apart(sensed[500:501], imagined[500:501])[0]
  assert got[1]["trace"][0][1] == (pytest.approx(-f[0], abs=1e-3),
                                   pytest.approx(-f[1], abs=1e-3))


# ---- the grading -------------------------------------------------------------------

@pytest.fixture(scope="module")
def drawn():
  lid = ch.draw(3)
  return lid, im.Setting(chest_x=1.4, chest_y=-0.2, chest_yaw=0.3), -0.6


def test_the_reference_graded_is_the_reference_and_the_world_statically(drawn):
  lid, setting, pin = drawn
  doc = ch.reference_document(lid, (setting.chest_x, setting.chest_y), setting.chest_yaw,
                              pin=pin)
  g = em.grade(doc, lid, setting, pin)
  e = em.errors(g)
  assert g["model"]["joint"] == "hinge" and g["model"]["type"] == "hinge"
  assert g["model"]["axisDeg"] == pytest.approx(0.0, abs=1e-6)
  assert abs(g["model"]["hingeAlongMm"]) < 1e-6 and abs(g["model"]["hingeUpMm"]) < 1e-6
  for k in ("static", "moment", "friction", "damping", "release", "holdNm"):
    v = e["vsReference"][k]
    assert np.allclose(v, 0.0, atol=1e-6), (k, v)
  # statically the language says the chest exactly: what it cannot say is
  # the catch's falloff, its contact and its armature
  assert np.allclose(e["language"]["static"], 0.0, atol=1e-6)
  assert e["language"]["moment"] == pytest.approx(0.0, abs=1e-9)
  assert all(s > 0 for s in g["truth"]["static"]), "the lid shuts at every angle swept"


def test_a_model_whose_lid_has_no_joint_is_graded_absent_never_zero(drawn):
  lid, setting, pin = drawn
  doc = ch.reference_document(lid, (setting.chest_x, setting.chest_y), setting.chest_yaw,
                              pin=pin)
  doc["joints"], doc["catches"] = [], []
  g = em.grade(doc, lid, setting, pin)
  assert g["model"] is None and g["lidPart"] == "lid"
  assert all(v is None for v in em.errors(g)["vsWorld"].values())


def test_the_fitters_own_instrument_is_the_reference_with_its_lids_dynamics_unknown(drawn):
  lid, setting, pin = drawn
  t = em.reference_template(lid, setting, pin, hinge_off_mm=(0.0, 5.0))
  # every number of the lid's the robot cannot see, the hidden weight's mass
  # among them: left true, the fitter alone was handed it
  assert lid.lump_kg > 0
  assert t.names == ["lid.mass", "lump.mass", "hinge.stiffness", "hinge.slack",
                     "hinge.damping", "hinge.friction", "hinge.release"]
  ref = ch.reference_document(lid, (setting.chest_x, setting.chest_y), setting.chest_yaw,
                              pin=pin)
  moved = next(j for j in t.raw["joints"] if j["id"] == "hinge")["at"]
  was = next(j for j in ref["joints"] if j["id"] == "hinge")["at"]
  assert moved[2] - was[2] == pytest.approx(5.0) and moved[:2] == pytest.approx(was[:2])


def test_an_interval_is_a_median_and_its_bootstrap_and_a_share_wilsons():
  iv = em.median_interval([1.0, 2.0, 3.0, None, 4.0, 5.0])
  assert iv["n"] == 5 and iv["median"] == 3.0 and iv["lo"] <= 3.0 <= iv["hi"]
  assert em.median_interval([None]) is None
  sh = em.share_interval([True] * 8 + [False] * 2)
  assert sh["share"] == 0.8 and 0.49 < sh["lo"] < 0.8 < sh["hi"] < 0.95


def test_the_arms_force_row_by_row_is_its_force_at_each_row():
  from pluggybot.legs import arm as am
  from pluggybot.legs.model import CHOSEN
  rng = np.random.default_rng(0)
  qs, qf = rng.uniform(1.5, 2.5, 20), rng.uniform(0.2, 0.9, 20)
  ts, te = rng.normal(size=20), rng.normal(size=20)
  rows = am.tool_forces(CHOSEN.arm, qs, qf, ts, te)
  for i in range(20):
    assert rows[i] == pytest.approx(am.tool_force(CHOSEN.arm, qs[i], qf[i], ts[i], te[i]))


# ---- the wire ----------------------------------------------------------------------

def test_the_ghost_is_the_models_parts_and_motion_and_no_number_of_its_dynamics():
  # What the site draws over the real object: its parts, where it drew them,
  # and how its imagination moved them along the probe -- the robot's
  # model, with none of the numbers a fit found or the world hides.
  from pluggybot.imagination import ghost
  from pluggybot.telemetry.protocol import IMAGINED_EVENT_TYPES
  doc = sc.parse_template(BOX).fill([2.0, 0.3, 0.05, 0.04])
  n = 300
  readings = Readings(t=np.arange(1, n + 1) * 0.002, sensed=np.zeros((n, 4)),
                      joints={"hinge": np.linspace(0.0, 1.0, n)})
  msg = ghost.message(doc, readings, robot="pluggybot", t=40.0, t0=10.0, dt=0.002, what="a box")
  assert msg["type"] in IMAGINED_EVENT_TYPES and json.loads(json.dumps(msg)) == msg
  assert [p["id"] for p in msg["parts"]] == ["base", "lid"]
  assert msg["parts"][1]["size"] == [0.2, 0.3, 0.012] and msg["rest"]["lid"][:3] == [0.6, 0.0, 0.146]
  replay = msg["replay"]
  assert replay["parts"] == ["lid"] and replay["t0"] == 10.0 and replay["hz"] == ghost.HZ
  every = round(1 / (ghost.HZ * 0.002))
  assert len(replay["frames"]) == math.ceil(n / every)
  a = float(np.linspace(0.0, 1.0, n)[(len(replay["frames"]) - 1) * every])
  assert replay["frames"][-1][0][3:] == pytest.approx([math.cos(a / 2), 0, math.sin(a / 2), 0],
                                                      abs=1e-4)

  def keys(o):
    if isinstance(o, dict):
      return set(o) | set().union(*(keys(v) for v in o.values()))
    if isinstance(o, list):
      return set().union(*(keys(v) for v in o)) if o else set()
    return set()
  assert not keys(msg) & {"mass", "stiffness", "slack", "damping", "friction", "release"}


def test_a_ghosts_cylinder_is_turned_as_a_scenes_is():
  # The ghost holds the scene's turn itself (the fence: importing the
  # scene's module loads the house), so the two are pinned equal.
  from pluggybot.imagination import ghost
  from pluggybot.telemetry import scene as ts
  assert ghost.AXIS_FIX == pytest.approx(ts.AXIS_FIX)


def test_a_scene_carries_no_hidden_weight():
  # Where the chest's hidden weight sits is a hidden parameter, and a scene
  # is on the wire: the weight is drawn with no alpha, which a scene omits.
  import mujoco
  from pluggybot.telemetry.scene import scene_dict
  spec = mujoco.MjSpec.from_string('<mujoco><worldbody><geom name="floor" type="plane" '
                                   'size="5 5 0.1"/></worldbody></mujoco>')
  ch.attach_chest(spec, ch.Lid(lump_kg=0.1, lump_at=0.7), (1.0, 0.0), 0.0, tags=False)
  scene = scene_dict(spec.compile(), "a_chest")
  names = [g.get("name") or "" for b in scene["bodies"] for g in b["geoms"]]
  assert "chest_lid" in names and not [n for n in names if "lump" in n]


def test_what_the_jaws_felt_is_the_torques_less_the_arms_own_weight_after_the_fact():
  # Off a record alone -- its readings, its encoders, its start's tilt and
  # the friction it calibrated: an arm holding still with nothing in its
  # jaws felt nothing, its own weight a few newtons at the tool.
  from pluggybot.imagination.compile import compile_scene
  from pluggybot.imagination.rollout import rollout
  from pluggybot.legs.imagined import felt
  rec = hold(400)
  readings = rollout(compile_scene(sc.parse(CABINET), carrying="module_claw"), rec)
  rec = Record(start=rec.start, dt=rec.dt, commands=rec.commands, sensed=readings.sensed)
  f = felt(rec, (0.1, 0.12))
  # every row, the ends too: smoothed over a zero pad, a still arm read 45
  # rad/s at either end and lost its friction there, 0.6 N at the tool
  assert np.abs(f).max() < 0.05
  from pluggybot.legs import arm as am
  from pluggybot.legs.model import CHOSEN
  raw = am.tool_forces(CHOSEN.arm, rec.sensed[:, 2], rec.sensed[:, 3], rec.sensed[:, 0],
                       rec.sensed[:, 1])
  assert np.abs(raw[100:].mean(axis=0)).max() > 1.0, "the premise: the arm weighs"


def test_a_picture_is_exposed_as_the_colour_sensor_would():
  from pluggybot.legs import probe as pr
  dark = np.full((40, 60, 3), 30, dtype=np.uint8)
  dark[10:30, 20:40] = 70
  import io
  from PIL import Image
  out = np.asarray(Image.open(io.BytesIO(pr.exposed(pr.jpeg(dark)))))
  assert out.min() < 15 and out.max() > 240
