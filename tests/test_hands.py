"""The humanoid's hands in a procedure, and practice graded by the world (issue
#493, stages 2 and 3): the vocabulary as stated priors -- the hands' verbs and
sensors through the language's registries, none of them knowing the chest --
the arms' friction calibrated once and kept, a library that admits a
procedure only on its grader's pass, and the practice loop that shows a mind
how its procedure ran and the world's grade, never the grade's secrets.

Every test here is a direct call or the stub body: the flights that measure
the reference and the practice are `scripts/attempts.py`'s.
"""

import json
import math
import re
from types import SimpleNamespace as NS

import numpy as np
import pytest

from pluggybot.body import StubBody
from pluggybot.evaluation import attempts as at
from pluggybot.evaluation import practice as ep
from pluggybot.humanoid import friction
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import parts as hp
from pluggybot.mind import practice as mp
from pluggybot.procedure import axes, lang
from pluggybot.procedure import steps as st
from pluggybot.procedure.library import Library, LibraryRefused

from test_lid import lift, run, stub_world


# ---- the arms' friction, calibrated once and kept --------------------------------


def test_the_arms_friction_is_kept_as_calibrated_and_matches_their_motors_sheets():
  """The robot's own calibration (`humanoid_spike.py --friction --write`)
  against what its motors' sheets give the model (ours, never the
  robot's): a stale table -- a motor changed, the file left -- fails."""
  table = friction.load()
  assert set(table) == {f"{s}_{j}" for s in hm.SIDES for j in hm.ARM}
  for name, row in table.items():
    sheet = hp.friction_nm(hm.CHOSEN.motor(name))
    assert row["coulomb"] == pytest.approx(sheet, rel=0.1), name
    assert abs(row["viscous"]) < 0.1, name


def test_friction_is_taken_off_only_a_joint_that_turns():
  table = {"left_elbow": {"coulomb": 0.1, "viscous": 0.02}}
  k = hm.ARM.index("elbow")
  qd = np.zeros(len(hm.ARM))
  qd[k] = 0.5
  assert friction.taken(table, "left", qd)[k] == pytest.approx(0.1 + 0.01)
  assert friction.taken(table, "left", -qd)[k] == pytest.approx(-0.11)
  # held still, its friction is anything up to its dry friction: none is taken
  qd[k] = friction.MOVING_RAD_S / 2
  assert not friction.taken(table, "left", qd).any()
  # ...and a joint with no row, or the other side, has none taken off
  qd[k] = 0.5
  assert not friction.taken(table, "right", qd).any()


def test_a_single_speed_cannot_tell_dry_friction_from_viscous_and_two_can():
  """The calibration sweeps at two speeds: at one, the two terms are one
  column and their split is anything."""
  dry, visc = 0.3, 0.05
  two = [(s * v, dry * s + visc * s * v, v) for v in (0.15, 0.45) for s in (1, -1)
         for _ in range(20)]
  got = friction.fit(two)
  assert got["coulomb"] == pytest.approx(dry) and got["viscous"] == pytest.approx(visc)
  one = [(s * 0.3, dry * s + visc * s * 0.3, 0.3) for s in (1, -1) for _ in range(20)]
  a = np.column_stack([np.sign([o[0] for o in one]), [o[0] for o in one]])
  assert np.linalg.matrix_rank(a) == 1


# ---- the vocabulary: the hands' verbs and sensors, through the registries -------


def test_a_hands_world_calls_its_own_verbs_and_a_served_world_is_unchanged():
  facts = st.hand_facts(pull=True)
  table = st.language(facts)
  assert set(table) == set(facts.verbs)
  assert st.language(None) is st.VERBS
  # an unknown verb is refused naming the verbs this body has, never the
  # served robot's
  with pytest.raises(st.Refused) as e:
    lang.compile_procedure("def a():\n    fetch(\"module_claw\")\n", st.hand_facts())
  assert "have: wait, move, spot, reach" in str(e.value) and "drive_to" not in str(e.value)
  # ...and the compliant pull is stage 3's: stage 2's vocabulary has none
  with pytest.raises(st.Refused):
    lang.compile_procedure("def a():\n    pull(\"left\", 0, 0, 1, 10, 0.1)\n", st.hand_facts())
  assert "pull" not in st.hand_facts().verbs and "pull.moved" not in st.hand_facts().sensors


def test_no_hand_verb_knows_the_chest():
  """Ben's condition on call 2: every verb generic, for any handle and any
  drawer -- a grip only shuts and says whether it holds, a hand is moved by
  its own verbs, a tag is found where it is -- and no default is the demo's
  grip (half a cube in, the fingers 25 deg down, the jaws across)."""
  for v in st.HANDS_VERBS.values():
    text = v.doc.lower()
    for word in ("lid", "chest", "knob", "hinge", "53", "drop handle"):
      assert not re.search(rf"\b{word}\b", text), (v.name, word)
    for name, arg in v.args.items():
      if name in ("fx", "fy", "fz", "jx", "jy", "jz"):
        assert arg.default == 0.0, (v.name, name)


def test_the_hands_axes_and_sensors_are_registered_as_a_body_registers_its_own():
  assert set(axes.BODY_AXES["humanoid"]) == {f"{s}_{j}" for s in hm.SIDES for j in hm.ARM}
  for n in axes.BODY_AXES["humanoid"]:
    lo, hi = hm.joint_range(n)
    assert (axes.AXES[n].lo, axes.AXES[n].hi) == (float(lo), float(hi))
  facts = st.hand_facts(pull=True)
  assert set(facts.sensors) <= set(axes.SENSORS)
  assert set(facts.axes) == set(axes.BODY_AXES["humanoid"])


class _Life:
  """A procedure's life on the stub body (`ProcedureLife`'s shape)."""

  def __init__(self, body):
    self.body, self.model, self.data = body, body.model, body.data
    self.reports, self.said = {}, []
    self.step_now = self.step_until = None
    self.aborting = False

  def interrupted(self):
    return False

  def _say(self, text, detail="", **_):
    self.said.append(text)


def _stub():
  return StubBody(*_bare())


def _bare():
  import mujoco
  m = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom name="floor" type="plane" size="3 3 0.1"/></worldbody></mujoco>')
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  return m, d


def _fly(source: str, body=None, pull: bool = True) -> dict:
  body = body or _stub()
  facts = st.hand_facts(pull=pull)
  proc = lang.compile_procedure(source, facts)
  return body.run(lang.run_procedure_routine(_Life(body), proc, facts))


def test_a_reach_that_does_not_hold_says_the_bodys_own_why():
  class Short(StubBody):
    def reach_routine(self, side, pos, rot=None, stiffness=1.0, patience=20.0):
      return False
      yield

    def hand_why(self, side):
      return "stopped short"
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n', Short(*_bare()))
  step = ran["steps"][-1]
  assert not step["ok"] and step["why"] == "stopped short"
  assert step["reason"] == st.REACH_WHY["stopped short"].format(hand="left")


def test_a_verbs_report_is_read_as_sensors_and_none_before_it_runs():
  """`pull.*` is the last pull's report; read before any pull, the step
  fails and says so, never a 0."""
  ran = _fly('def a():\n    x = read("pull.moved")\n    wait(1)\n')
  assert "failedAt" in ran and "pull" in ran["steps"][-1]["reason"]
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n    pull("left", 0, 0, 1, 10, 0.1)\n'
             '    m = read("pull.moved")\n    z = read("pull.dz")\n')
  assert ran["ok"] and ran["locals"]["m"] == pytest.approx(0.1)
  assert ran["locals"]["z"] == pytest.approx(1.0)


def test_a_hand_is_told_and_read_in_its_stance_frame():
  """A place a procedure gives is ahead / left / up of the stance, wherever
  the body stands and whichever way it faces."""
  body = _stub()
  body.start_at(2.0, 1.0, math.pi / 2)
  ran = _fly('def a():\n    reach("left", 0.3, 0.1, 0.9)\n    x = read("left.x")\n'
             '    y = read("left.y")\n', body)
  assert ran["locals"]["x"] == pytest.approx(0.3) and ran["locals"]["y"] == pytest.approx(0.1)
  wx, wy, _ = body.hand_at("left")
  assert (wx, wy) == pytest.approx((2.0 - 0.1, 1.0 + 0.3))


# ---- a library that admits only what its grader passed --------------------------


SOURCE = 'def opener():\n    wait(1)\n'


def test_a_library_admits_a_procedure_only_on_its_graders_pass():
  lib = Library(st.hand_facts())
  with pytest.raises(LibraryRefused, match="no grader's verdict"):
    lib.admit("opener", SOURCE, {"ok": True})
  failed, passed = [], []
  run(lift(up_to=0.5), sealed=failed)
  run(lift(), sealed=passed)
  assert not failed[0].ok and passed[0].ok
  with pytest.raises(LibraryRefused, match="did not pass it"):
    lib.admit("opener", SOURCE, failed[0])
  assert lib.names() == ()
  lib.admit("opener", SOURCE, passed[0])
  assert lib.runnable() == ("opener",)
  assert [r["verb"] for r in lib.refusals] == ["admit", "admit"]


# ---- practice: a mind shown how its procedure ran and the world's grade ----------


class Mind:
  """A stand-in mind: the answers it is given, in turn, and what it was
  asked (`messages.create`'s seam)."""

  def __init__(self, *sources):
    self.messages, self.sources, self.asked = self, list(sources), []

  def create(self, **kw):
    self.asked.append(kw)
    text = json.dumps({"think": "a try", "source": self.sources.pop(0)})
    return NS(content=[NS(type="text", text=text)], usage=NS(input_tokens=100, output_tokens=40))


def _practice(*sources, attempts=2):
  facts = st.hand_facts()
  mind = Mind(*sources)
  pr = mp.Practice(mind, "a-mind", facts, ep.JOB.format(limit=30))
  rows = ep.practise(0, pr, attempts, picture=False, budget_s=30.0,
                     build=lambda t: stub_world(t))
  return mind, pr, rows


def test_a_try_is_flown_graded_and_shown_back_and_a_refusal_goes_back_first():
  mind, pr, rows = _practice("def a():\n    nonsense()\n", 'def a():\n    wait(1)\n',
                             'def b():\n    wait(2)\n')
  # the refused source went back with its reasons, costing tokens, no try
  assert [t.kind for t in pr.turns] == ["first", "repair", "after"]
  assert "nonsense" in mind.asked[1]["messages"][-1]["content"]
  assert [r["attempt"] for r in rows] == [1, 2]
  first = rows[0]
  assert first["process"] == "practice" and first["skill"] == "a"
  assert first["tokens"] == 2 * 140 and first["steps"] == 1
  assert [s["verb"] for s in first["practice"]["ran"]["steps"]] == ["wait"]
  assert not first["verdict"]["ok"]
  # the next turn is how the try went and the world's grade
  report = mind.asked[2]["messages"][-1]["content"]
  assert "TRY 1" in report and first["verdict"]["reason"] in report
  assert "line 2 wait: ok" in report and "(1 left)" in report


def test_a_try_is_shown_its_grade_never_its_secrets():
  """The catch's force is the grader's secret (`lid`'s): on the row under
  `secret`, never in what the mind is shown."""
  _, _, rows = _practice('def a():\n    wait(1)\n', attempts=1)
  row = rows[0]
  text = mp.report_text(1, row["practice"]["ran"], row["verdict"], row["simS"], row["cut"], 0)
  for name, value in row["secret"].items():
    assert value is not None and f"{value}" not in text, name
  assert "That was your last try." in text


def test_the_practice_prompt_prescribes_nothing():
  """Its rules say what the body and the language are, never how to open
  anything; the job is told as its grader measures it."""
  for pull in (False, True):
    text = mp.system_prompt(st.hand_facts(pull=pull)).lower()
    for word in ("lid", "chest", "knob", "hinge", "arc", "handle", "box", "drawer", "door"):
      assert not re.search(rf"\b{word}\b", text), word
  assert "both grippers start shut" in mp.system_prompt(st.hand_facts()).lower()
  job = ep.JOB.format(limit=300)
  assert "57 degrees" in job and "1.0 rad/s" in job and "tag 53" in job and "300 s" in job
