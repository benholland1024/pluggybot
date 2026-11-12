"""The humanoid's hands in a procedure, and practice graded by the world (issue
#493, stages 2 and 3): the vocabulary as stated priors -- the hands' verbs and
sensors through the language's registries, none of them knowing the chest --
the arms' friction calibrated once and kept, a library that admits a
procedure only on its grader's pass, and the practice loop that shows a mind
how its procedure ran and the world's grade, never the grade's secrets.

Every test here is a direct call or the stub body: the flights that measure
the reference and the practice are `scripts/attempts.py`'s.
"""

import functools
import json
import math
import re
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import pytest

from pluggybot import lifecycle as lc
from pluggybot.body import StubBody, body_for
from pluggybot.evaluation import practice as ep
from pluggybot.home import storeroom as sr
from pluggybot.humanoid import friction
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import parts as hp
from pluggybot.mind import practice as mp
from pluggybot.perception import tags
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
  assert set(table) == {f"{s}_{j}" for s in hm.SIDES for j in hm.ARM} | set(hm.WAIST)
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
  facts = st.hand_facts()
  table = st.language(facts)
  assert set(table) == set(facts.verbs)
  assert st.language(None) is st.VERBS
  # an unknown verb is refused naming the verbs this body has, never the
  # served robot's
  with pytest.raises(st.Refused) as e:
    lang.compile_procedure("def a():\n    fetch(\"module_claw\")\n", st.hand_facts())
  assert "have: wait, move, spot, watch, reach" in str(e.value) and "drive_to" not in str(e.value)
  # ...and the hands' vocabulary is ONE (#525): every hand verb, a verb as
  # an if's test and the maths wherever there are hands, `measure` alone an
  # arm's to take away
  assert set(st.HAND_VERBS) <= set(facts.verbs) and "measure" in facts.verbs
  assert "pull.moved" in facts.sensors and facts.verb_tests and facts.math
  bare = st.hand_facts(measure=False)
  assert "measure" not in bare.verbs and set(st.HAND_VERBS) <= set(bare.verbs)


def test_a_pull_asks_no_more_than_the_standing_body_holds_and_says_so():
  """#544 (Ben, 2026-10-25): `pull` asks at most what the standing body
  holds at a hand, 20 N -- held 3 s at a counter's height it stood to 30 N
  pulling back and 20 pushing away -- and its text says so, and that
  pushing away holds around two thirds of it. More is refused before a
  step runs, as any bound is."""
  pull = st.language(st.hand_facts())["pull"]
  assert pull.args["force"].hi == st.PULL_MOST_N == 20.0
  assert "your body holds a pull of 20 N at a hand, and pushing away from you around " \
         "two thirds of that" in " ".join(pull.doc.split())
  lang.compile_procedure('def a():\n    pull("left", -1, 0, 0, 20, 0.1)\n', st.hand_facts())
  with pytest.raises(st.Refused, match="force=21.0 is above 20.0"):
    lang.compile_procedure('def a():\n    pull("left", -1, 0, 0, 21, 0.1)\n', st.hand_facts())


def test_no_hand_verb_knows_the_chest():
  """Ben's condition on call 2: every verb generic, for any handle and any
  drawer -- a grip only shuts and says whether it holds, a hand is moved by
  its own verbs, a tag is found where it is -- and no default is the demo's
  grip (half a cube in, the fingers 25 deg down, the jaws across)."""
  for v in st.HANDS_VERBS.values():
    text = v.doc.lower()
    for word in ("lid", "chest", "knob", "hinge", "53", "drop handle", "drawer", "door",
                 "lever", "cupboard", "cabinet", "cube", "mat", "54", "55", "56", "57"):
      assert not re.search(rf"\b{word}\b", text), (v.name, word)
    for name, arg in v.args.items():
      if name in ("fx", "fy", "fz", "jx", "jy", "jz"):
        assert arg.default == 0.0, (v.name, name)


def test_the_hands_axes_and_sensors_are_registered_as_a_body_registers_its_own():
  assert set(axes.BODY_AXES["humanoid"]) == {f"{s}_{j}" for s in hm.SIDES for j in hm.ARM}
  for n in axes.BODY_AXES["humanoid"]:
    lo, hi = hm.joint_range(n)
    assert (axes.AXES[n].lo, axes.AXES[n].hi) == (float(lo), float(hi))
  facts = st.hand_facts()
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


def _fly(source: str, body=None, measure: bool = True) -> dict:
  body = body or _stub()
  facts = st.hand_facts(measure=measure)
  proc = lang.compile_procedure(source, facts)
  return body.run(lang.run_procedure_routine(_Life(body), proc, facts))


def test_a_reach_that_does_not_hold_says_the_bodys_own_why():
  class Short(StubBody):
    def reach_routine(self, side, pos, rot=None, stiffness=1.0, patience=20.0, refuse=False):
      return False
      yield

    def hand_why(self, side):
      return "stopped short"
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n', Short(*_bare()))
  step = ran["steps"][-1]
  assert not step["ok"] and step["why"] == "stopped short"
  assert step["reason"] == st.REACH_WHY["stopped short"].format(hand="left")


@pytest.mark.parametrize("why", ["held", "fell"])
def test_a_withdraw_is_ok_only_once_its_arm_is_home(why):
  """#539: `withdraw` said ok as the arm's walk home began, the arm still
  seconds from home; it is ok once the body says home, else says why not."""
  class Away(StubBody):
    def release_routine(self, side, refuse=False):
      return why
      yield
  source = 'def a():\n    reach("left", 0.3, 0.2, 0.9)\n    withdraw("left")\n'
  step = _fly(source, Away(*_bare()))["steps"][-1]
  assert step["verb"] == "withdraw" and not step["ok"] and step["why"] == why
  assert step["reason"] == st.WITHDRAW_WHY[why].format(hand="left")
  assert _fly(source)["ok"]


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
  rows = ep.practice_on(0, pr, attempts, picture=False, budget_s=30.0,
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
  assert first["process"] == "practice" and first["skill"] == "a-mind"
  assert first["practice"]["procedure"] == "a"
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


def test_a_try_its_time_cut_is_shown_what_it_ran_never_no_procedure():
  """A procedure its try's limit cuts mid-verb ran, and the mind is shown
  what it ran and where the time ran out -- never "You had no procedure
  that compiled" (#530's review: the record was kept only when the run
  returned, and a cut run never does)."""
  _, _, rows = _practice('def a():\n    wait(1)\n    wait(60)\n', attempts=1)
  row = rows[0]
  assert row["cut"] and row["steps"] == 2
  assert [s["verb"] for s in row["practice"]["ran"]["steps"]] == ["wait"]
  text = mp.report_text(1, row["practice"]["ran"], row["verdict"], row["simS"], row["cut"], 0)
  assert "no procedure that compiled" not in text and "line 2 wait: ok" in text
  assert "was at line 3, wait, when the time ran out" in text
  assert "The try's time ran out after 30 s." in text


def test_the_practice_prompt_prescribes_nothing():
  """Its rules say what the body and the language are, never how to open
  anything; the job is told as its grader measures it."""
  for facts in (st.hand_facts(measure=False), st.hand_facts(), st.hand_facts(step=False)):
    text = mp.system_prompt(facts).lower()
    for word in ("lid", "chest", "knob", "hinge", "arc", "handle", "box", "drawer", "door"):
      assert not re.search(rf"\b{word}\b", text), word
  assert "both grippers start shut" in mp.system_prompt(st.hand_facts()).lower()
  job = ep.JOB.format(limit=300)
  assert "57 degrees" in job and "1.0 rad/s" in job and "tag 53" in job and "300 s" in job
  assert "26 mm" in job


# ---- the reference: the hand turns with the lid past the handle's stop -----------


def test_the_reference_turns_the_hand_with_the_lid_only_past_the_handles_stop():
  """A drop handle hangs free on its pin until its arm lies on the lid's
  front edge, 57 deg up: the arc keeps the hand's turn to `FREE_RAD`, the
  knob moving as the pin does, and turns it as the lid turns past it."""
  from pluggybot.evaluation import opener as op
  from pluggybot.evaluation.opener import Guess, rot_about
  g = Guess(knob=(0.35, 0.2, 0.81), yaw=0.0, top=0.97, hinge=(0.63, 0.2, 0.97),
            pin=(0.36, 0.2, 0.87))
  start, rot = np.array(g.knob), op.grip_turn(g, 0.0)
  ups, downs = op.arc(g, start, rot, top=op.TOP)
  rises = list(np.arange(op.STEP, op.TOP - 1e-9, op.STEP)) + [op.TOP]
  assert len(ups) == len(rises)
  for (p, r), s in zip(ups, rises):
    turned = max(0.0, s - op.FREE_RAD)
    assert r == pytest.approx(rot_about(g.axis, turned) @ rot, abs=1e-9)
    pin = np.array(g.hinge) + rot_about(g.axis, s) @ (np.array(g.pin) - g.hinge)
    assert p == pytest.approx(pin + rot_about(g.axis, turned) @ (start - g.pin), abs=1e-9)
  # ...and back down the same way, to under the grader's shut line
  assert downs[-1][0] == pytest.approx(ups[0][0] + (ups[0][0] - ups[1][0]), abs=0.02)
  assert downs[-1][1] == pytest.approx(rot, abs=1e-9)


def test_the_body_members_the_reference_leans_on_are_the_stubs_too():
  """`gaze_routine`, `hand_why` and a reach or a carry along a path: every
  body answers them (the stub at once)."""
  body = _stub()
  body.run(body.gaze_routine(0.25))
  assert body.hand_why("left") == ""
  assert body.run(body.reach_routine("left", (0.3, 0.2, 0.9), path=[(0.3, 0.2, 1.0)],
                                     turns=[np.eye(3)]))
  got = body.run(body.move_hand_routine("left", (0.3, 0.2, 1.1), path=[(0.3, 0.2, 1.0)],
                                        turns=[np.eye(3), np.eye(3)]))
  assert got["why"] == "arrived"


def test_the_jobs_time_limit_is_its_tasks_and_the_grader_reads_it():
  """Each set-out's own (`attempts.LIMIT_S`: the humanoid's, three times its
  reference's slowest pass at the counter), handed to the grader as
  `limitS`."""
  from pluggybot.challenge import lid
  from pluggybot.evaluation import attempts as at
  task = at.Task(chest=0)
  assert task.limit_s == at.LIMIT_S[("humanoid", "counter")] < at.BUDGET_S
  sealed = []
  row = run(lift(), sealed=sealed)
  # ...judged unrounded and shown to a tenth (`lid.SHOWN`)
  assert sealed[0].ok and sealed[0].metrics["limitS"] == round(task.limit_s, lid.SHOWN["limitS"])
  assert row["run"]["budgetS"] == task.limit_s
  m = dict(sealed[0].metrics, limitS=1.0)
  ok, _, why = lid.eval_lid(m)
  assert not ok and "past the 1 s the job allows" in why


# ---- a hand that is out: the rules its carries were debugged into --------------


def _bare_reach():
  """A `Reach` with no world: its own methods over state a test sets, its
  postures recorded rather than asked of a policy."""
  from pluggybot.humanoid import arms as ha
  from pluggybot.humanoid import reach as rc
  r = object.__new__(rc.Reach)
  r.self_model = ha.SelfModel()
  r.rest = {s: np.zeros(len(hm.ARM)) for s in hm.SIDES}
  r.phase = {s: "rest" for s in hm.SIDES}
  r.goal = {s: None for s in hm.SIDES}
  r.cmd = {s: None for s in hm.SIDES}
  r.q_to = {s: np.zeros(len(hm.ARM)) for s in hm.SIDES}
  r.q_cmd = {s: np.zeros(len(hm.ARM)) for s in hm.SIDES}
  r.miss, r._off = {s: 0.0 for s in hm.SIDES}, {s: 0.0 for s in hm.SIDES}
  r.slack = {s: True for s in hm.SIDES}
  r.speed = {s: rc.HAND_SPEED for s in hm.SIDES}
  r.pulls = {s: None for s in hm.SIDES}
  r.report = {s: None for s in hm.SIDES}
  r.posture = np.zeros(3)
  r.postures = []
  r.hold_posture = lambda height, pitch, waist=0.0: r.postures.append((height, pitch, waist))
  r.waist_cmd, r.waist_speed = np.zeros(len(hm.WAIST)), np.zeros(len(hm.WAIST))
  # ...its waist's driver commanded as recorded, never a world's
  r.waist_sent = []
  r.waist = NS(command=lambda target, feed: r.waist_sent.append(np.array(target)))
  r.grippers = {s: _Grip() for s in hm.SIDES}
  r.homes, r._planner = {s: [] for s in hm.SIDES}, None
  r.outs = {s: [] for s in hm.SIDES}
  # ...and no other robot's body for its ways to keep off (#678)
  r.keep_out, r.kept = [], {s: None for s in hm.SIDES}
  r.allowed, r.left = {s: [] for s in hm.SIDES}, {s: [] for s in hm.SIDES}
  r.looks, r.looking = {s: 0 for s in hm.SIDES}, {s: False for s in hm.SIDES}
  r.sensed = np.full(len(hm.JOINT_NAMES), np.nan)
  # the whole body's reading off: the senses here are a test's stubs, not a
  # body's (`_feel`'s own tests turn it on) -- and the body taken as still
  # (#631: `arms.Motion`'s own tests are `test_felt`'s)
  r.standing = False
  r.arm_force = {s: np.zeros(3) for s in hm.SIDES}
  r.motion = NS(read=lambda sm, qd, stand: None, sample=lambda *a: None)
  r.stand, r.d = NS(), None
  return r


def _standing_reach():
  """A bare reach on the robot's model of itself at the stand, its arms'
  rest the stand's and commanded there."""
  r = _bare_reach()
  q = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
  r.self_model.pose(q, np.array([0.0, 0.0, -1.0]))
  r.rest = {s: q[r.self_model.arm[s]].copy() for s in hm.SIDES}
  r.q_cmd = {s: r.rest[s].copy() for s in hm.SIDES}
  return r


class _Grip:
  """A gripper as a bare reach reads it: holding what a test says."""

  holding = False

  def held(self):
    return self.holding

  def sample(self):
    pass


def _turned(deg):
  from pluggybot.evaluation.opener import rot_about
  return rot_about(np.array([0.0, 1.0, 0.0]), math.radians(deg))


@pytest.mark.parametrize("phase", ["joints", "move"])
def test_an_arm_with_no_line_out_withdraws_home_in_joint_space(phase):
  """`move` on an arm's joint, then `withdraw`: there is no line to come
  out along, and unhandled the withdraw waited out the attempt -- nor for an
  arm still walking out to its pre-grasp, whose command is a past hold's;
  and the posture goes back to the stand."""
  from pluggybot.humanoid.arms import HandGoal
  r = _bare_reach()
  r.set_joints("left", r.rest["left"] + 0.3)
  if phase == "move":
    r.cmd["left"] = HandGoal(np.array([0.3, 0.2, 0.9]), None, 1.0)
    r.goal["left"], r.phase["left"] = HandGoal(np.array([0.4, 0.2, 0.9]), None, 1.0), "move"
  r.retreat("left")
  assert r.phase["left"] == "home" and r.goal["left"] is None
  assert np.allclose(([r.q_to["left"]] + r.homes["left"])[-1], r.rest["left"])
  assert r.postures == [(0.0, 0.0, 0.0)]


#: Where the arm's walk home began in #539's flights, raised before its
#: shoulder: the right's (pitch, roll, yaw, elbow, wrist pitch and roll),
#: and the left's mirror.
RAISED = np.array([-1.72, -1.44, 0.95, 1.03, 1.27, -0.05])


def _of(side, q_right):
  """The right arm's joints `q_right` for `side`: the left's mirror them
  (`reach.seed`, its own inverse)."""
  from pluggybot.humanoid import reach as rc
  return q_right if side == "right" else rc.seed("right", q_right)


def _posed_reach(pose: str):
  """`_standing_reach`, or posed as a reach's statue at the stand
  (`reach.plan`): its trunk 2.7 cm further ahead, the hanging hand past the
  toes."""
  from pluggybot.humanoid import reach as rc
  r = _standing_reach()
  if pose == "statue":
    rc._posed(r.self_model, 0.0, 0.0, np.array([hm.stand_qpos().get(n, 0.0)
                                                for n in hm.JOINT_NAMES]))
  return r


def _sent_home(r, side, q0):
  """The arm commanded at `q0` and sent home: the joint targets it walks."""
  r.set_joints(side, q0)
  r.q_cmd[side] = np.asarray(q0, dtype=float).copy()
  r.retreat(side)
  return [r.q_to[side]] + r.homes[side]


def _clear_of_counter(sm, side, q0, ways, front=0.279, top=0.72, turned=0.0) -> float:
  """The least distance from a counter as wide as the world -- its front
  edge `front` ahead of the soles' middle, its top `top` up (chest 0's),
  the robot `turned` deg toward it on the arm's side (its edge nearer
  there) -- to the arm's solid parts, m (negative: into it), as the arm
  walks in joint space from `q0` through `ways`: each capsule's axis less
  its radius, each box's corners."""
  import itertools
  import mujoco
  worst, prev = math.inf, np.asarray(q0, dtype=float)
  for w in ways:
    for t in np.linspace(0.0, 1.0, 41)[1:]:
      sm._set_arm(side, prev + (w - prev) * t)
      for part in ("upper_arm", "forearm", "gripper", "jaw_a_pad", "jaw_b_pad"):
        g = sm.m.geom(f"{side}_{part}").id
        pos, rot, size = sm.d.geom_xpos[g], sm.d.geom_xmat[g].reshape(3, 3), sm.m.geom_size[g]
        if sm.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
          pts, r = [pos + a * size[1] * rot[:, 2] for a in np.linspace(-1, 1, 5)], size[0]
        else:
          pts, r = [pos + rot @ (np.array(c) * size)
                    for c in itertools.product((-1, 1), repeat=3)], 0.0
        for p in pts:
          edge = front - math.tan(math.radians(turned)) * abs(p[1])
          off = (edge - p[0], p[2] - top)          # outside by either
          d = math.hypot(max(off[0], 0), max(off[1], 0)) if max(off) > 0 else max(off)
          worst = min(worst, d - r)
    prev = w
  return worst


@pytest.mark.parametrize("pose", ["stand", "statue"])
@pytest.mark.parametrize("side", hm.SIDES)
def test_an_arm_raised_over_a_counter_goes_home_at_its_side_never_down_onto_it(side, pose):
  """#539: straight home in joint space, a hand raised before its shoulder
  brought a jaw down onto the counter's front edge and rested there 1.3 s;
  judged on the robot's model of itself (straight home, the arm dips ahead
  of the robot's front), it rolls out to its side first and is lowered
  there, clear of any counter so far ahead. ⚠ As a reach's statue stands
  too: its hanging hand past the toes, judged from the toes alone straight
  home tied the side way and took it."""
  r = _posed_reach(pose)
  q0 = _of(side, RAISED)
  ways = _sent_home(r, side, q0)
  assert np.allclose(ways[-1], r.rest[side])
  sm = r.self_model
  assert _clear_of_counter(sm, side, q0, [r.rest[side]]) < 0, "the premise: straight onto it"
  assert _clear_of_counter(sm, side, q0, ways) > 0.05


#: Where the left arm's walk home began in the capability check's drawer
#: (#525, set-out 3), the drawer pulled 17 cm out toward the robot.
PULLED_CLOSE = np.array([0.75, 0.18, 0.11, 1.14, 1.18, 1.76])


def test_an_arm_none_of_it_ahead_of_the_robots_front_goes_straight_home():
  """What the hand worked over lies under it ahead of the robot's front, so
  an arm with none of it there worked over nothing there. Judged as if all
  under it were solid, the check's drawer, pulled close, took the side way:
  its roll-out swept a jaw into the drawer's face and pushed it 4 cm back
  in, and the job failed."""
  from pluggybot.humanoid import reach as rc
  r = _standing_reach()
  sm = r.self_model
  ahead = rc.front(sm, "left", r.rest["left"])
  sm._set_arm("left", PULLED_CLOSE)
  assert not (rc.outline(sm, "left")[:, 0] > ahead).any(), "the premise: none of it ahead"
  _sent_home(r, "left", PULLED_CLOSE)
  assert r.homes["left"] == [] and np.allclose(r.q_to["left"], r.rest["left"])


#: Where the cube check's left arm began its way home at set-out 0, the cube
#: set down (#560): its grip had left its roll at 2.75 rad.
CUBE_DOWN = np.array([0.9433, 2.7543, 0.0736, 2.0231, -0.5382, 1.0559])


@pytest.mark.parametrize("start", ["cube", "raised"])
def test_an_arm_comes_home_behind_a_counter_it_stands_turned_to(monkeypatch, start):
  """#560: the cube check stood 3-4 cm from its counter's face, turned 6
  deg to it; lowered in the hanging arm's line, the side way came down 55
  cm out, where the edge was nearer, and dragged a jaw over its top for
  10 s, the robot turning 20 deg. Its roll comes in pitched back
  (`HOME_BACK`). On the statue, the edge 2 cm nearer: the live arm hangs
  that much further forward."""
  from pluggybot.humanoid import reach as rc
  q0 = CUBE_DOWN if start == "cube" else _of("left", RAISED)
  r = _standing_reach()
  ways = _sent_home(r, "left", q0)
  assert _clear_of_counter(r.self_model, "left", q0, ways, front=0.13, turned=6.0) > 0.005
  monkeypatch.setattr(rc, "HOME_BACK", 0.0)
  r = _standing_reach()
  ways = _sent_home(r, "left", q0)
  assert _clear_of_counter(r.self_model, "left", q0, ways, front=0.13, turned=6.0) < -0.01, \
    "the premise: lowered in its hanging line, onto it"


@pytest.mark.parametrize("side", hm.SIDES)
def test_the_side_way_rolls_an_arm_out_to_its_own_side(side):
  """From a counter's height (the grip 0.30 m ahead, 0.25 out, 0.85 up), its
  roll inside `HOME_ROLL`: rolled out to it, on its own side, held there as
  its other joints come home; from #539's raised arm, rolled out further
  already, its roll kept."""
  from pluggybot.humanoid import reach as rc
  k = hm.ARM.index("shoulder_roll")
  lo, hi = hm.joint_range(f"{side}_shoulder_roll")
  outward = 1.0 if hi > -lo else -1.0
  r = _standing_reach()
  y = 0.25 if side == "left" else -0.25
  q0, miss, _ = rc.solve_seeded(r.self_model, side, np.array([0.30, y, 0.85]), None)
  assert miss < 0.01 and abs(q0[k]) < rc.HOME_ROLL
  ways = _sent_home(r, side, q0)
  assert len(ways) == 4
  assert ways[0][k] * outward == pytest.approx(rc.HOME_ROLL) and ways[1][k] == ways[0][k]
  q0 = _of(side, RAISED)
  ways = _sent_home(_standing_reach(), side, q0)
  assert ways[0][k] == pytest.approx(q0[k]) and ways[1][k] == pytest.approx(q0[k])


def test_an_arm_no_lower_on_its_way_home_ahead_of_its_toes_goes_straight_there():
  """The side way is a detour (its sweep out to the side, 3 s): a hand low
  ahead of the robot's front, whose way home rises, walks straight home."""
  from pluggybot.humanoid import reach as rc
  r = _standing_reach()
  sm = r.self_model
  ahead = rc.front(sm, "right", r.rest["right"])
  q0, miss, _ = rc.solve_seeded(sm, "right", np.array([0.12, -0.2, 0.6]), None)
  sm._set_arm("right", q0)
  assert miss < 0.01 and (rc.outline(sm, "right")[:, 0] > ahead).any(), "the premise: some ahead"
  _sent_home(r, "right", q0)
  assert r.homes["right"] == [] and np.allclose(r.q_to["right"], r.rest["right"])


@pytest.mark.parametrize("straight,aside,taken", [(0.5, 0.0, 1), (5.0, 8.0, 1), (5.0, 0.0, 4)])
def test_the_side_way_is_taken_only_past_the_margin_and_where_it_dips_less(
    monkeypatch, straight, aside, taken):
  """`way_home`'s rule over the dips its judge gives each way, in
  `HOME_DIP_M`s: straight home unless that dips more than the margin, and
  the side way only where it dips less -- a detour that buys nothing is
  not taken."""
  from pluggybot.humanoid import reach as rc
  monkeypatch.setattr(rc, "dip", lambda sm, side, q0, ways, *a:
                      (straight if len(ways) == 1 else aside) * rc.HOME_DIP_M)
  assert len(_sent_home(_standing_reach(), "right", RAISED)) == taken


def test_an_arm_is_home_once_its_rest_is_commanded_and_its_drivers_read_it_there():
  """`withdraw` is ok on `Reach.home` (#539): the rest commanded -- never an
  arm still on its way, whose command says it hangs -- and every joint read
  within `HOME_RAD` of it."""
  from pluggybot.humanoid import reach as rc
  r = _standing_reach()
  arm = r.self_model.arm["left"]
  r.sensed[arm] = r.rest["left"]
  assert r.home("left")
  r.sensed[arm[3]] += 2 * rc.HOME_RAD
  assert not r.home("left")
  r.sensed[arm[3]] = r.rest["left"][3]
  _sent_home(r, "left", r.rest["left"] + 0.3)
  assert r.phase["left"] == "home" and not r.home("left")


def test_a_second_retreat_partway_home_leaves_the_way_be():
  """`to_walk_routine` sends every arm not at rest home again: one partway
  along its way goes on, never planned afresh from where it is."""
  r = _standing_reach()
  way = [q.copy() for q in _sent_home(r, "right", RAISED)]
  r.q_cmd["right"] = (RAISED + r.rest["right"]) / 2
  r.retreat("right")
  got = [r.q_to["right"]] + r.homes["right"]
  assert len(got) == len(way) == 4 and all(np.allclose(a, b) for a, b in zip(way, got))


def _stepped(r):
  """A bare reach whose loop (`Reach._update`) runs: its senses, its trim and
  its drivers stubbed out."""
  from types import SimpleNamespace
  n = len(hm.JOINT_NAMES)
  r.tau, r.qd, r.friction = np.zeros(n), np.zeros(n), {}
  r.force = {s: np.zeros(3) for s in hm.SIDES}
  r._pose = lambda arms=None: None
  r._trim = lambda dt: None
  r.arms = {s: SimpleNamespace(command=lambda *a: None) for s in hm.SIDES}
  return r


def test_the_walk_home_walks_every_target_in_turn_then_rests():
  """The loop walks the way home target by target, `home` until the last
  arrives -- its command at the rest -- and only then `rest`."""
  r = _stepped(_standing_reach())
  way = [q.copy() for q in _sent_home(r, "right", RAISED)]
  seen = [r.q_to["right"].copy()]
  for _ in range(2000):
    r._update(0.01)
    if not np.allclose(r.q_to["right"], seen[-1]):
      seen.append(r.q_to["right"].copy())
    if r.phase["right"] != "home":
      break
  assert r.phase["right"] == "rest" and np.allclose(r.q_cmd["right"], r.rest["right"])
  assert len(seen) == len(way) == 4 and all(np.allclose(a, b) for a, b in zip(seen, way))


def test_the_joints_an_arm_is_judged_home_by_are_its_drivers_reading():
  """`Reach.sensed` is what the drivers read -- never the command the model
  is posed at after (#539: aliased to it, home was the command's arrival
  again)."""
  from types import SimpleNamespace
  r = _standing_reach()
  read = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES]) + 0.2
  r.d = None
  r.stand = SimpleNamespace(
    joints=SimpleNamespace(read=lambda d: (read.copy(), np.zeros_like(read), np.zeros_like(read))),
    trunk=SimpleNamespace(gravity=lambda: np.array([0.0, 0.0, -1.0])))
  r._pose(r.q_cmd)
  assert np.allclose(r.sensed, read)


def test_the_way_home_is_judged_as_the_robot_stands_and_moves_nothing_of_its_model():
  """Judged on the planner's model posed as the live one stands, whatever the
  planner was left at, and the live model left as it was."""
  from pluggybot.humanoid import reach as rc
  want = [q.copy() for q in _sent_home(_standing_reach(), "right", RAISED)]
  r = _standing_reach()
  rc._posed(r._planner_model(), -0.25, 1.1, np.array([hm.stand_qpos().get(n, 0.0)
                                                      for n in hm.JOINT_NAMES]))
  live = r.self_model.q.copy(), r.self_model.d.qpos.copy()
  got = _sent_home(r, "right", RAISED)
  assert len(got) == len(want) and all(np.allclose(a, b) for a, b in zip(want, got))
  assert np.array_equal(r.self_model.q, live[0]) and np.array_equal(r.self_model.d.qpos, live[1])


#: A table's edge at the robot's front (#560's set-out, 13 cm out): 7 mm
#: ahead of the hanging grip, whose top is at the top's underside.
TABLE_EDGE = 0.116


def _over_table(side):
  """A reach's goal over that table where the set-out's cup stands, 0.42 m
  ahead and 0.30 to its side, fingers down at 45 deg as the cup is taken."""
  from pluggybot.humanoid.arms import HandGoal, frame
  return HandGoal(np.array([0.415, 0.304 if side == "left" else -0.304, 0.78]),
                  frame((0.707, 0.0, -0.707), (0.0, 1.0, 0.0)))


def _pre_grasp(r, side):
  from pluggybot.humanoid import reach as rc
  return r._planned(side, _over_table(side), rc.APPROACH_M, rc.REACH_POSTURES, True)[0][1]


def test_an_arm_reaching_over_a_table_at_its_front_goes_out_at_its_side():
  """#560: standing 13 cm from a 0.72 m table's edge, the left arm's
  straight way out in joint space to a pre-grasp over it drove its forearm
  and gripper into the table at 353-359 N; judged as the way home is, from
  its other end (`way_out`), it pitches back, rolls out to its side, goes
  out with its roll held, then in, clear of a table whose edge is at the
  robot's front."""
  r = _standing_reach()
  q_pre = _pre_grasp(r, "left")
  assert r.reach("left", _over_table("left")) and r.phase["left"] == "move"
  ways = [r.q_to["left"]] + r.outs["left"]
  sm = r.self_model
  assert len(ways) == 4 and np.allclose(ways[-1], q_pre)
  assert _clear_of_counter(sm, "left", r.rest["left"], [q_pre], front=TABLE_EDGE) < -0.05, \
    "the premise: straight into it"
  assert _clear_of_counter(sm, "left", r.rest["left"], ways, front=TABLE_EDGE) > 0.01


def test_a_hand_already_out_reaches_on_straight():
  """The side way is an arm's from its own side: one out ahead of the
  robot's front walks straight to its next pre-grasp, sweeping nothing it
  worked over aside."""
  from pluggybot.humanoid import reach as rc
  r = _standing_reach()
  q0, miss, _ = rc.solve_seeded(r.self_model, "right", np.array([0.35, -0.25, 0.85]), None)
  assert miss < 0.01
  r.q_cmd["right"] = q0
  q_pre = _pre_grasp(_standing_reach(), "right")
  assert len(r.way_out("right", q_pre)) == 1


def test_the_way_out_walks_every_target_in_turn_and_a_stop_ends_it():
  """The loop walks the way out target by target to the pre-grasp, the
  approach waiting on its last (the posture held back here); and a way out
  ended -- withdrawn, stopped (`halt`), or an arm moved joint by joint --
  walks none of what was left of it."""
  from types import SimpleNamespace
  from pluggybot.humanoid.arms import HandGoal
  r = _stepped(_standing_reach())
  r.stand = SimpleNamespace(settled=False)
  q_pre = _pre_grasp(r, "right")
  way = r.way_out("right", q_pre)
  r.goal["right"] = HandGoal(np.zeros(3), None, 1.0)
  r.q_to["right"], r.outs["right"], r.phase["right"] = way[0].copy(), [q.copy() for q in way[1:]], "move"
  seen = [r.q_to["right"].copy()]
  for _ in range(2000):
    r._update(0.01)
    if not np.allclose(r.q_to["right"], seen[-1]):
      seen.append(r.q_to["right"].copy())
  assert r.phase["right"] == "move" and np.allclose(r.q_cmd["right"], q_pre)
  # (a leg whose ends are one pose -- a roll already out -- changes no target)
  legs = [q for k, q in enumerate(way) if not k or not np.allclose(q, way[k - 1])]
  assert len(seen) == len(legs) >= 3 and all(np.allclose(a, b) for a, b in zip(seen, legs))
  for end in ("retreat", "halt", "set_joints"):
    r = _standing_reach()
    r.stand = SimpleNamespace(trunk=SimpleNamespace(gravity=lambda: np.array([0.0, 0.0, -1.0])))
    r.goal["right"] = HandGoal(np.zeros(3), None, 1.0)
    r.q_to["right"], r.outs["right"], r.phase["right"] = way[0].copy(), [way[1].copy()], "move"
    if end == "set_joints":
      r.set_joints("right", r.rest["right"])
    else:
      getattr(r, end)("right", *(("until",) if end == "halt" else ()))
    assert r.outs["right"] == [], end


def test_a_carry_that_stops_says_how_short_it_fell_and_holds_the_turn_it_was_asked():
  from pluggybot.humanoid.arms import HandGoal
  r = _bare_reach()
  asked, sagged = np.eye(3), _turned(9.0)
  r.cmd["left"] = HandGoal(np.zeros(3), asked.copy(), 1.0)
  r.goal["left"] = HandGoal(np.array([0.10, 0.0, 0.0]), asked.copy(), 1.0)
  r.pulls["left"] = {"start": np.zeros(3), "t": 1.0, "peak": 0.0}
  r.phase["left"] = "carry"
  h = np.array([0.03, 0.0, 0.0])
  r._end("left", "out of reach", h, sagged)
  assert r.report["left"]["shortM"] == pytest.approx(0.07)
  assert np.allclose(r.goal["left"].pos, h) and np.allclose(r.cmd["left"].pos, h)
  assert np.allclose(r.cmd["left"].rot, asked) and np.allclose(r.goal["left"].rot, asked)


def test_a_carry_keeps_the_turn_it_was_asked_and_a_path_of_turns_keeps_its_posture():
  from pluggybot.humanoid.arms import HandGoal
  r = _bare_reach()
  asked, sagged, h = np.eye(3), _turned(9.0), np.array([0.35, 0.2, 0.9])
  r.cmd["left"] = HandGoal(h.copy(), asked.copy(), 1.0)
  r.goal["left"] = HandGoal(h.copy(), asked.copy(), 1.0)
  r.phase["left"] = "hold"
  r.grip_now = lambda side: (h.copy(), sagged)
  planned = []
  r.posture_for = lambda side, pos, rot=None: planned.append(rot) or (0.0, 0.0, 0.0)
  r.path_for = lambda side, poses, rot=None: planned.append(rot) or (0.0, 0.0, 0.0)
  assert r.carry("left", h + [0.0, 0.0, 0.05])
  assert np.allclose(planned[-1], asked) and np.allclose(r.cmd["left"].rot, asked)
  assert not r.pulls["left"]["fixed"]
  # ...and one given its turns alone is a path: its posture held to its end
  assert r.carry("left", h + [0.0, 0.0, 0.05], path=[], turns=[asked])
  assert r.pulls["left"]["fixed"]


def test_a_two_jaw_grip_is_planned_half_a_turn_round_where_asked_and_never_when_told_not(
    monkeypatch):
  from pluggybot.humanoid import reach as rc
  from pluggybot.humanoid.arms import HandGoal
  asked = _turned(10.0)

  def plan(side, goal, model=None, postures=rc.REACH_POSTURES, approach=rc.APPROACH_M, **kw):
    return None if np.allclose(goal.rot, asked) else ((0.0, 0.0, 0.0), np.zeros(6), np.zeros(6),
                                                      True)
  monkeypatch.setattr(rc, "plan", plan)
  r = _bare_reach()
  r.slack = {s: True for s in hm.SIDES}
  assert r.reach("left", HandGoal(np.array([0.35, 0.2, 0.9]), asked, 1.0))
  assert np.allclose(r.goal["left"].rot, asked @ np.diag([-1.0, -1.0, 1.0]))
  assert not _bare_reach().reach("left", HandGoal(np.array([0.35, 0.2, 0.9]), asked, 1.0),
                                 flip=False)


def test_the_statue_stands_back_with_its_arm_held_out():
  """The planner's statue balanced with the arm where it is going: held out
  forward, its pelvis stands back of the one with arms hanging, as the
  standing policy's does (3-7 cm in front of the robot, balanced hanging)."""
  from pluggybot.humanoid import arms as ha
  from pluggybot.humanoid import reach as rc
  sm = ha.SelfModel()
  q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
  rc._posed(sm, 0.0, 0.0, q0)
  hanging = float(sm.d.qpos[0])
  rc._posed(sm, 0.0, 0.0, q0, rc._arm("left", np.array([-1.4, 0.1, 0.0, 0.3, 0.0, 0.0])))
  assert float(sm.d.qpos[0]) < hanging - 0.005


def test_the_hands_force_stays_bounded_where_the_arm_is_straight():
  """Read off the motors through the grip's Jacobian, a straight arm's
  force blew up on a few hundredths of a N*m: the directions the arm
  cannot tell apart are left out (`arms.FORCE_SIGMA`)."""
  from pluggybot.humanoid import arms as ha
  sm = ha.SelfModel()
  q = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
  sm.pose(q, np.array([0.0, 0.0, -1.0]))
  sm._set_arm("left", np.zeros(len(hm.ARM)))
  noise = np.array([0.03, -0.02, 0.02, -0.03, 0.02, -0.01])
  assert float(np.linalg.norm(sm.hand_force("left", sm.gravity("left") + noise))) < 5.0


def test_a_shut_grippers_word_waits_until_its_jaws_have_closed_or_held():
  """A knob swings off a pad and back as the jaws close: read at once, the
  gripper said it held nothing (`body.GRIP_SETTLE_S`)."""
  from pluggybot.humanoid import body as hb
  from pluggybot.humanoid.arms import HELD_GAP_M
  clock = {"t": 0.0}

  class Jaws:
    shut_asked = False

    def shut(self, force=None):
      self.shut_asked = True

    def open(self, width=None):
      self.shut_asked = False

    def held(self):
      return self.shut_asked and clock["t"] >= 1.0

    def read(self):
      return (2 * HELD_GAP_M, 0.0)
  b = object.__new__(hb.HumanoidBody)
  b.mission = NS(to_stand_routine=lambda: iter(()),
                 reach=NS(grippers={"left": Jaws()}, ease=lambda side: False))

  def hold(seconds):
    clock["t"] += seconds
    return
    yield
  b.hold_routine = hold
  routine = b.grip_routine("left", True)
  with pytest.raises(StopIteration) as done:
    next(routine)
  assert done.value.value is True


def test_a_hand_pressed_on_what_it_met_is_eased_before_its_jaws_shut():
  """#560: a cube's grip point, the fingers' middle, held 2-3 cm past where
  their tips met its board, the arm pressing them on it at 40-120 N; the
  jaws dragged across it as they shut and the feet swivelled 4-6 deg.
  Held -- holding, or stopped on its way, pressed hardest -- and pushing on
  along its fingers past `EASE_N`, the hand is held where its senses put
  it, its command walked there at `HAND_SPEED` (a setpoint is ramped);
  under it, or on its way, nothing moves."""
  from pluggybot.humanoid import reach as rc
  from pluggybot.humanoid.arms import HandGoal, frame
  r = _bare_reach()
  rot = frame((0.5, 0.0, -0.866), (0.0, 1.0, 0.0))
  goal = np.array([0.32, 0.17, 0.78])
  at = goal - 0.025 * (-rot[:, 2])                 # its tips met the board 2.5 cm short
  for phase, push, eased in (("hold", rc.EASE_N + 5.0, True), ("blocked", rc.EASE_N + 5.0, True),
                             ("hold", rc.EASE_N - 5.0, False), ("approach", rc.EASE_N + 5.0, False),
                             ("carry", 40.0, False)):
    r.phase["left"], r.speed["left"] = phase, rc.HOLD_SPEED
    r.cmd["left"], r.goal["left"] = HandGoal(goal.copy(), rot, 1.0), HandGoal(goal.copy(), rot, 1.0)
    r.sensed_grip = {"left": at.copy(), "right": None}
    r.force = {"left": push * -rot[:, 2], "right": np.zeros(3)}
    assert r.ease("left") is eased, (phase, push)
    assert np.allclose(r.goal["left"].pos, at if eased else goal) and np.allclose(r.cmd["left"].pos, goal)
    if eased:
      assert r.phase["left"] == "hold" and np.allclose(r.goal["left"].rot, rot)
      # ...its command walked there one command at a time, never put there
      r._line("left", 0.01)
      step = float(np.linalg.norm(r.cmd["left"].pos - goal))
      assert step == pytest.approx(rc.HAND_SPEED * 0.01, rel=1e-6)


@pytest.mark.parametrize("closed", [True, False])
def test_a_shut_eases_its_hand_first_and_an_open_does_not(closed):
  from pluggybot.humanoid import body as hb
  order = []

  class Jaws:
    def shut(self, force=None):
      order.append("shut")

    def open(self, width=None):
      order.append("open")

    def held(self):
      return True

    def read(self):
      return (0.02, 50.0)
  b = object.__new__(hb.HumanoidBody)
  b.mission = NS(to_stand_routine=lambda: iter(()),
                 reach=NS(grippers={"left": Jaws()}, ease=lambda side: order.append("ease")))
  b.hold_routine = lambda seconds: iter(())
  b.run = None
  for _ in b.grip_routine("left", closed):
    pass
  assert order == (["ease", "shut"] if closed else ["open"])


def test_the_mind_is_told_the_time_limit_its_try_is_cut_at():
  from pluggybot.evaluation import attempts as at
  task = at.Task(chest=0, body="humanoid", mount="counter")
  assert f"{task.limit_s:.0f} s at most" in ep.job_for(task)
  facts = st.hand_facts()
  told_300 = mp.Practice(Mind('def a():\n    wait(1)\n'), "a-mind", facts, ep.JOB.format(limit=300))
  with pytest.raises(ValueError, match="another time limit"):
    ep.practice_on(0, told_300, 1, picture=False, budget_s=30.0, build=lambda t: stub_world(t))


def test_the_verbs_speeds_and_slew_are_the_reachs_own():
  """The procedure's verbs state the reach's speeds and its joints' rate,
  copied where the loop's side may not import the body's routines: a
  number moved on one side is moved on both."""
  import inspect
  from pluggybot.body import Body
  from pluggybot.humanoid import reach as rc
  assert st.HAND_V[1] == rc.HOLD_SPEED and st.HAND_V[2] == rc.HAND_SPEED
  assert axes.HUMANOID_SLEW == rc.JOINT_RATE
  for name in ("move_hand_routine", "pull_routine"):
    assert inspect.signature(getattr(Body, name)).parameters["speed"].default == rc.HOLD_SPEED


def _out_hand(r, h, rot):
  """`r`'s left hand out at `h` (stance frame), carried toward 5 cm up, its
  model posed at the stand."""
  from pluggybot.humanoid.arms import HandGoal
  q = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
  r.self_model.pose(q, np.array([0.0, 0.0, -1.0]))
  r.cmd["left"] = HandGoal(h.copy(), rot.copy(), 1.0)
  r.goal["left"] = HandGoal(h + [0.0, 0.0, 0.05], rot.copy(), 1.0)
  r.phase["left"] = "carry"
  r.pulls["left"] = {"start": h.copy(), "t": 0.0, "ways": [], "fixed": True, "limit": None,
                     "peak": 0.0}


def test_a_carried_hand_holds_still_while_its_body_takes_a_new_posture():
  """Carried on while the trunk moved, a lid was jerked down at 1.7-2.3
  rad/s and up to 81 deg: the command waits for the posture."""
  r = _bare_reach()
  h, rot = np.array([0.30, 0.20, 0.95]), np.eye(3)
  _out_hand(r, h, rot)
  r.pulls["left"]["postured"] = 0.0
  r._posture_held = lambda: False
  for _ in range(20):
    r._held("left", 0.01, h, rot)
  assert np.allclose(r.cmd["left"].pos, h)
  r._posture_held = lambda: True
  for _ in range(20):
    r._held("left", 0.01, h, rot)
  assert r.cmd["left"].pos[2] > h[2] + 0.005


def test_a_solve_that_diverged_is_never_commanded():
  """A miss of 10-15 cm commanded swung an arm round a chest's knob and out
  of it (`reach.WILD_M`): the arm keeps its last command."""
  from pluggybot.humanoid import reach as rc
  r = _bare_reach()
  h, rot = np.array([0.30, 0.20, 0.95]), np.eye(3)
  _out_hand(r, h, rot)
  r.cmd["left"].pos = np.array([1.5, 0.2, 0.95])        # out of any reach
  r.goal["left"].pos = r.cmd["left"].pos.copy()
  before = r.q_cmd["left"].copy()
  r._held("left", 0.01, h, rot)
  assert np.allclose(r.q_cmd["left"], before)
  assert rc.WILD_M < 1.0


# ---- feedback a procedure can debug by, and what a robot keeps --------------------


def test_a_reach_out_of_reach_says_whether_the_other_hand_could():
  """18 of 27 first tries of #493's flight took the right hand to a knob on
  the robot's left: the refusal says what the body's own model says of the
  other hand, asked the same pose -- either way."""
  body = _stub()
  body.out_of_reach = {"right"}
  step = _fly('def a():\n    reach("right", 0.3, 0.2, 0.9)\n', body)["steps"][-1]
  assert step["reason"] == (st.REACH_WHY["out of reach"].format(hand="right")
                            + "; your left hand could reach it as asked")
  body.out_of_reach = {"right", "left"}
  step = _fly('def a():\n    reach("right", 0.3, 0.2, 0.9)\n', body)["steps"][-1]
  assert step["reason"].endswith("; nor could your left hand, as asked")
  # ...and where the TURN is what is out of reach, it says so, off the same plan
  body = _stub()
  body.unturned = {"left", "right"}
  step = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9, fx=1, jz=1)\n', body)["steps"][-1]
  assert step["reason"] == (st.REACH_WHY["out of reach"].format(hand="left")
                            + " turned as asked; turned another way it can (leave fx..jz at 0"
                            " and your arm finds a turn); nor could your right hand as asked,"
                            " though turned another way it can")
  step = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n', body)["steps"][-1]
  assert step["ok"]
  # ...and so of the OTHER hand: "nor could your left hand, as asked", of a
  # right hand turned toward a knob on the left, was read as out of reach
  # standing, and the robot stepped beside the box 59 tries of 60
  body = _stub()
  body.out_of_reach, body.unturned = {"right"}, {"left"}
  step = _fly('def a():\n    reach("right", 0.3, 0.2, 0.9, fx=1, jz=1)\n', body)["steps"][-1]
  assert step["reason"] == (st.REACH_WHY["out of reach"].format(hand="right")
                            + "; nor could your left hand as asked, though turned another way"
                            " it can (leave fx..jz at 0 and your arm finds a turn)")
  # ...and a hand that stopped short is not out of reach: nothing is said of the other
  class Short(StubBody):
    def reach_routine(self, side, pos, rot=None, stiffness=1.0, patience=20.0, refuse=False):
      return False
      yield

    def hand_why(self, side):
      return "stopped short"
  step = _fly('def a():\n    reach("right", 0.3, 0.2, 0.9)\n', Short(*_bare()))["steps"][-1]
  assert "other" not in step["reason"] and "left" not in step["reason"]


def test_a_pull_takes_a_way_of_any_length():
  """Bounded to 1 a component, two tries ended at a computed 1.03 and 1.001:
  the way is any length, scaled to one, and only no way at all is refused."""
  facts = st.hand_facts()
  assert st.check_arg(st.HANDS_VERBS["pull"], "dz", 1.03, facts) == []
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n    pull("left", 0, 0, 2.5, 10, 0.1)\n'
             '    z = read("pull.dz")\n    m = read("pull.moved")\n')
  assert ran["ok"] and ran["locals"]["z"] == pytest.approx(1.0)
  assert ran["locals"]["m"] == pytest.approx(0.1)
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n    pull("left", 0, 0, 0, 10, 0.1)\n')
  assert "give the pull a way" in ran["steps"][-1]["reason"]


def test_a_pull_hands_the_body_its_force_and_give_and_one_that_never_ran_reads_nothing():
  """What a pull asks reaches the body as asked; and with a verb an `if`'s
  test, the last pull's readings would read as a failed one's (#525)."""
  body = _stub()
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n'
             '    pull("left", 0, 0, -1, 4, 0.1, give=0)\n    pull("left", 0, 0, 1, 12, 0.05)\n',
             body)
  assert ran["ok"] and [p[2:] for p in body.pulled] == [(4.0, 0.1, False), (12.0, 0.05, True)]
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n    pull("left", 0, 0, 1, 10, 0.05)\n'
             '    withdraw("left")\n    if not pull("left", 0, 0, 1, 10, 0.05):\n'
             '        m = read("pull.moved")\n', _stub())
  assert "failedAt" in ran and "pull" in ran["steps"][-1]["reason"]


class _PullEnds(StubBody):
  """A stub whose pull ends the way it is told to, as `Body.pull_routine`
  reports it."""

  ends = "distance"

  def pull_routine(self, side, direction, limit_n, distance_m, speed=0.06, patience=30.0,
                   give=True):
    why = self.ends.pop(0) if isinstance(self.ends, list) else self.ends
    return {"why": why, "seconds": 0.5, "moved": 0.02, "along": 0.02, "dx": 0.0,
            "dy": 0.0, "dz": 0.02, "direction": [0.0, 0.0, 1.0], "force": float(limit_n)}
    yield


@pytest.mark.parametrize("ends, reads", [
  ("distance", True), ("held fast", True), ("let go", True), ("out of reach", True),
  ("feet moved", True), ("out of time", True), ("until", True),
  ("not out", False), ("fell", False)])
def test_a_pull_reads_false_only_where_its_text_says(ends, reads):
  """#650: read "ok when it pulled", one robot built its loops on a pull
  held fast reading false and the other was right that it reads ok. Only a
  hand that was not out, or a fall, reads false, however else it stopped (a
  way all 0 is the test above's, an argument refused the next one's); and
  `pull.held` says which of its stops was held fast."""
  body = _PullEnds(*_bare())
  body.ends = ends
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n    p = 0\n'
             '    if pull("left", 0, 0, 1, 10, 0.1):\n        p = 1\n'
             '        h = read("pull.held")\n', body)
  assert ran["locals"]["p"] == (1 if reads else 0), ran
  if reads:
    assert ran["locals"]["h"] == (1.0 if ends == "held fast" else 0.0), ran


def test_a_pull_that_reads_false_leaves_no_reading_of_its_own():
  """As its text says: after a pull that went ok, one that read false leaves
  none of its readings, `pull.held` ending the run as `pull.moved` always
  has (#525) -- the first pull's would read as this one's. (A pull that
  fell ends the run there, read by a test or not: the test below.)"""
  body = _PullEnds(*_bare())
  body.ends = ["held fast", "not out"]
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n    pull("left", 0, 0, 1, 10, 0.1)\n'
             '    was = read("pull.held")\n    p = 1\n'
             '    if not pull("left", 0, 0, 1, 10, 0.1):\n        p = 0\n'
             '    h = read("pull.held")\n', body)
  assert ran["locals"]["was"] == 1.0 and ran["locals"]["p"] == 0, ran
  assert ran["steps"][-1]["reason"] == "pull.held has no reading: your last `pull` failed", ran


@pytest.mark.parametrize("verb", ["reach", "pull"])
def test_a_hand_verb_that_fell_ends_the_run_though_an_if_tests_it(verb):
  """#671 (Luca, 11-08 15:29): `chest_free4`'s reach inside an `if` fell,
  read false, and the run went on -- its `withdraw` waited out the get-up,
  the topple and the death, and three more verbs ran on a dead robot. A
  fall ends the run as it fails a job's try, a test or not: the run stops
  at the verb that fell, in that verb's own words, no test counted read
  false for it, and History says "it stopped at line N: you fell...".
  Shown to fail without the fall's exception in `lang`: the `if` read
  false, and both withdraws ran."""
  class Falls(_PullEnds):
    ends = "fell"

    def reach_routine(self, side, pos, rot=None, stiffness=1.0, patience=20.0, path=None,
                      turns=None, refuse=False):
      if verb == "reach":
        self.falls += 1          # the fall watch's count, as the get-up takes the body
        return False
      return True
      yield

    def hand_why(self, side):
      return "fell" if verb == "reach" else ""

    def pull_routine(self, *a, **kw):
      self.falls += 1
      return (yield from super().pull_routine(*a, **kw))
  call = ('reach("left", 0.3, 0.2, 0.9)' if verb == "reach"
          else 'pull("left", 0, 0, 1, 10, 0.1)')
  source = ("def a():\n    n = 0\n"
            + ('    reach("left", 0.3, 0.2, 0.9)\n' if verb == "pull" else "")
            + f"    if not {call}:\n        withdraw(\"left\")\n        n = 1\n"
            "    withdraw(\"left\")\n    n = 2\n")
  ran = _fly(source, Falls(*_bare()))
  line = 3 if verb == "reach" else 4
  said = (st.REACH_WHY if verb == "reach" else st.MOVE_WHY)["fell"].format(hand="left")
  step = ran["steps"][-1]
  assert not ran["ok"] and ran["failedAt"] == len(ran["steps"]) - 1, ran
  assert (step["verb"], step["line"], step["reason"]) == (verb, line, said), step
  assert step["fell"] and "tested" not in step and ran["testsFalse"] == 0, step
  assert ran["locals"] == {"n": 0.0}, "a line past the fall ran"
  [history] = lc.procedure_outcome("a", ran)
  assert history.startswith(f"the procedure a did not finish -- it stopped at line {line}, "
                            f"{verb}: {said} ("), history


def test_a_hand_whose_move_a_death_ended_is_halted_where_it_is():
  """#671's second review: a death closes a hand verb's routine, but the
  reach's own move goes on, on the physics seam -- a pull went on pulling
  for a dead robot. The verb's hand is halted where it is (`Body.stop_hand`,
  as a stop condition halts one), and the run ends `died`. Shown to fail
  without the halt."""
  class Pulling(StubBody):
    def stop_hand(self, side):
      self.halted.append(side)
      return True

    def pull_routine(self, side, direction, limit_n, distance_m, speed=0.06, patience=30.0,
                     give=True):
      for n in range(200):
        if n == 50:
          self.life.dead = {"cause": "flat"}  # the death seam's, as the pull goes on
        yield self.STILL
      return {"why": "distance", "seconds": 0.4, "moved": 0.1, "along": 0.1, "dx": 0.0,
              "dy": 0.0, "dz": 0.1, "direction": [0.0, 0.0, 1.0], "force": 5.0}
  body = Pulling(*_bare())
  body.halted = []
  body.life = life = _Life(body)
  facts = st.hand_facts()
  proc = lang.compile_procedure('def a():\n    reach("left", 0.3, 0.2, 0.9)\n'
                                '    pull("left", 0, 0, 1, 10, 0.1)\n    wait(1)\n', facts)
  ran = body.run(lang.run_procedure_routine(life, proc, facts))
  assert ran["stopped"] == "died" and ran["steps"][-1]["verb"] == "pull", ran
  assert body.halted == ["left"], body.halted


def test_a_call_refused_before_it_ran_leaves_its_verb_no_reading():
  """#650's review: a computed argument out of its bounds reads false in an
  `if`'s test and the verb never runs (`lang`). What it last reported must
  not read as this call's -- a retry computes just such a distance, what is
  left of `pull.moved` -- and its stop condition's reading is a move's that
  ended otherwise, so reading it is not reading the last pull's."""
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n'
             '    pull("left", 0, 0, 1, 10, 0.1, until="time", above=1e9)\n'
             '    d = 0.1 - read("pull.moved")\n    p = 1\n'
             '    if not pull("left", 0, 0, 1, 10, d):\n        p = 0\n'
             '    s = read("pull.stopped")\n    m = read("pull.moved")\n')
  assert ran["locals"]["p"] == 0 and ran["locals"]["s"] == 0.0, ran
  assert ran["steps"][-1]["reason"] == "pull.moved has no reading: your last `pull` failed", ran
  # ...every verb's, not the pull's alone: a step's would be taken off a
  # place a second time
  ran = _fly('def a():\n    step(0.2)\n    d = 9 * read("step.dx")\n    p = 1\n'
             '    if not step(d):\n        p = 0\n    x = read("step.dx")\n')
  assert ran["locals"]["p"] == 0, ran
  assert ran["steps"][-1]["reason"] == "step.dx has no reading: your last `step` failed", ran
  # ...and a stop condition's reading is a move's that ended otherwise, off
  # a move that did stop on one; a verb that never read one gets none
  life = _Life(_stub())
  life.reports = {"pull": {"moved": 0.1}, "stopped": {"pull": 1.0, "reach": 1.0}}
  st.forget_readings(life, "pull")
  st.forget_readings(life, "spot")
  assert life.reports["pull"] == {axes.UNREAD: "your last `pull` failed"}
  assert life.reports["stopped"] == {"pull": 0.0, "reach": 1.0}


def test_a_read_after_a_spot_that_failed_says_so_at_its_line():
  """#686 (D5's table practice, set-out 5, tries 2 and 3): `spot(90)` ran
  and failed, then `ez = read("spot.z")` stopped the run, and the robot was
  told "stopped at line ?, expr: spot.z has no reading: `spot` has not
  run". A failed `spot` forgets its report, as it must, but every missing
  report was worded as a verb never run; and a fault in a plain statement
  carried no line. The read says the last `spot` failed, at its own line,
  in the practice's report and in History. Shown to fail, both halves, on
  the tree before it."""
  ran = _fly('def a():\n    if not spot(90):\n        pass\n    ez = read("spot.z")\n')
  step, said = ran["steps"][-1], "spot.z has no reading: your last `spot` failed"
  assert (step["verb"], step.get("line"), step["reason"]) == ("expr", 4, said), step
  report = mp.run_report(2, ran, {"ok": False, "reason": "no"}, 1.0, False)
  assert f"and stopped at line 4, expr: {said}." in report, report
  [history] = lc.procedure_outcome("a", ran)
  assert history.startswith("the procedure a did not finish -- it stopped at line 4, an "
                            f"expression: {said} ("), history


class _SpotsOnce(StubBody):
  """A stub whose `spot` sees its tag once, and then never again."""

  seen = 0

  def spot(self, tag, at_height=None):
    self.seen += 1
    return ({"x": 0.5, "y": 0.0, "z": 0.9, "range": 0.6, "normal": (1.0, 0.0, 0.0),
             "edge": (0.0, 1.0, 0.0)} if self.seen == 1 else None)


@pytest.mark.parametrize("verb, source, sensor", [
  # a `spot` that saw nothing, one handed no whole tag, and one refused
  # before it ran
  ("spot", '    if not spot(90):\n        pass\n', "spot.x"),
  ("spot", '    if not spot(90.5):\n        pass\n', "spot.range"),
  ("spot", '    t = 900\n    if not spot(t):\n        pass\n', "spot.nz"),
  # a `pull` with no hand out, a `step` and a `measure` refused
  ("pull", '    if not pull("left", 0, 0, 1, 10, 0.1):\n        pass\n', "pull.moved"),
  ("step", '    d = 9\n    if not step(d):\n        pass\n', "step.dx"),
  ("measure", '    x = 9\n    if not measure(x, 0):\n        pass\n', "object.top"),
])
def test_a_verb_whose_last_call_failed_reads_so_never_that_it_has_not_run(verb, source, sensor):
  """#686: a failure or a refusal forgets what its verb last reported --
  it would read as this call's -- and reading it says that last call
  failed, where every verb's said it had not run. A verb never called in
  the run still says so. Shown to fail on the words as they were."""
  ran = _fly(f'def a():\n{source}    v = read("{sensor}")\n')
  assert ran["steps"][-1]["reason"] == f"{sensor} has no reading: your last `{verb}` failed", ran
  assert _fly(f'def a():\n    wait(0)\n    v = read("{sensor}")\n')["steps"][-1]["reason"] == (
    f"{sensor} has no reading: `{verb}` has not run")


def test_a_spot_that_saw_and_then_failed_reads_as_failed_not_as_the_last_one():
  """#686, the lie itself: a `spot` that read, then one that failed -- the
  second forgets the first's reading, and says it was the one that failed,
  never that `spot` has not run."""
  ran = _fly('def a():\n    spot(53)\n    x = read("spot.x")\n    if not spot(53):\n'
             '        pass\n    y = read("spot.x")\n', _SpotsOnce(*_bare()))
  assert ran["locals"]["x"] == pytest.approx(0.5), ran
  assert ran["steps"][-1]["reason"] == "spot.x has no reading: your last `spot` failed", ran


def test_a_failed_calls_word_ends_with_its_procedure_and_a_last_report_does_not():
  """#686's second review: a served robot keeps its reports from one
  procedure to the next, and a `spot` that failed in one told the next,
  which never called it, "your last `spot` failed". The words a run's
  failed calls leave end with it, as its watch does: the next run's read
  says `spot` has not run in it. What a verb last reported is kept, as it
  always was. Shown to fail on the tree before it."""
  facts = st.hand_facts()
  body = _SpotsOnce(*_bare())
  life = _Life(body)
  for src, said in (('def a():\n    spot(53)\n', None),
                    ('def b():\n    wait(0)\n    x = read("spot.x")\n', 0.5),
                    ('def c():\n    if not spot(53):\n        pass\n', None),
                    ('def d():\n    wait(0)\n    x = read("spot.x")\n',
                     "spot.x has no reading: `spot` has not run")):
    ran = body.run(lang.run_procedure_routine(life, lang.compile_procedure(src, facts), facts))
    if isinstance(said, str):
      assert ran["steps"][-1]["reason"] == said, ran
    elif said is not None:
      assert ran["locals"]["x"] == pytest.approx(said), ran


def test_pull_says_when_it_reads_false_and_spot_which_face_of_a_block_it_reads():
  """#650: the words both robots asked about. `pull`'s text says when it
  reads false -- no "ok when it pulled", held fast being ok too -- and how
  it is told how far it went; `spot`'s and `watch`'s say which face of a
  small block they read. What is true of them is pinned beside: the test
  above, and the next one."""
  pull = " ".join(st.HANDS_VERBS["pull"].doc.split())
  assert "ok when it pulled" not in pull
  assert ("whether it ended held fast and the way it was going at its end ride `pull.*`, "
          "unless it reads false") in pull
  assert ("ok unless an argument was refused, your hand was not out, (dx, dy, dz) are all 0 "
          "or you fell") in pull
  assert "whatever else stopped it, held fast against something too, it is still ok" in pull
  assert "`pull.moved` says how far it went" in pull
  spot = " ".join(st.HANDS_VERBS["spot"].doc.split())
  assert ("A small block carries its tag on every face: the one read is the one you see "
          "most square-on, `spot.nx`..`nz` say which") in spot
  assert "another face, its middle up to the block's width from this one's" in spot
  watch = " ".join(st.HANDS_VERBS["watch"].doc.split())
  assert "the face that picture shows most square-on" in watch
  told = " ".join(mp.system_prompt(st.hand_facts()).split())
  for doc in (pull, spot, watch):
    assert doc in told
  assert "1 if the last `pull` ended held fast, as your hand felt it" in told


def _standing():
  """The robot's body standing at its keyframe, one `mj_forward`: its cameras
  placed as its model draws them."""
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  return model, data


def _picture(model, data, camera):
  """`camera`'s picture as the robot stands: how far below level its top and
  its bottom are and how far either side it goes, degrees, its place and
  turn."""
  i = model.camera(camera).id
  rot = data.cam_xmat[i].reshape(3, 3)
  down = math.degrees(math.asin(rot[2, 2]))  # a camera looks along its -z
  half = float(model.cam_fovy[i]) / 2
  side = math.degrees(math.atan(math.tan(math.radians(half)) * tags.PICTURE_W / tags.PICTURE_H))
  return {"top": down - half, "bottom": down + half, "side": side, "half": half,
          "at": data.cam_xpos[i].copy(), "rot": rot}


def _bearing(pic, up):
  """How far to the side of straight ahead `pic`'s right edge reaches at its
  top (`up` 1) or its bottom (-1), degrees, the robot facing along x as at
  its keyframe: its corner turned into the world by the camera's own turn,
  the picture being tilted, never the camera-frame half-angle."""
  x, y, _ = pic["rot"] @ np.array([math.tan(math.radians(pic["side"])),
                                   up * math.tan(math.radians(pic["half"])), -1.0])
  return math.degrees(math.atan2(abs(y), x))


def _in_picture(pic, point):
  """Whether `point`, in the world, is inside `pic` (a camera looks along
  its -z, its y up)."""
  x, y, z = pic["rot"].T @ (np.asarray(point, float) - pic["at"])
  tall, wide = math.tan(math.radians(pic["half"])), math.tan(math.radians(pic["side"]))
  return bool(z < 0 and abs(y / -z) <= tall and abs(x / -z) <= wide)


def test_spot_and_watch_say_which_camera_they_read_and_what_its_picture_takes_in():
  """#660: both robots took `spot` for short-sighted and went on pointing it
  at tags above its picture. `spot`'s and `watch`'s text name the colour
  camera and say what its picture takes in, in the model's numbers, and
  `look`'s and `find`'s name the other camera; nothing calls either of them
  the head camera any more."""
  model, data = _standing()
  colour = _picture(model, data, "color_eye")
  spot = " ".join(st.HANDS_VERBS["spot"].doc.split())
  assert "your colour camera" in spot and "head camera" not in spot
  assert f"runs from {colour['top']:.0f} to {colour['bottom']:.0f} degrees below level" in spot
  assert (f"out to {_bearing(colour, 1):.0f} degrees either side of where your head faces at "
          f"its top and more than {int(_bearing(colour, -1) // 10 * 10)} at its bottom") in spot
  assert _bearing(colour, 1) < _bearing(colour, -1), "the picture is tilted: wider at its bottom"
  assert "bowing only takes it lower" in spot
  # the picture's top in the stance frame, whose origin is the soles' middle
  soles = np.mean([data.geom_xpos[model.geom(f"{s}_sole").id] for s in hm.SIDES], axis=0)
  drop = math.tan(math.radians(colour["top"]))
  at_feet = float(colour["at"][2]) + drop * float(colour["at"][0] - soles[0])
  assert st.SPOT_TOP_AT_FEET_M == pytest.approx(at_feet, abs=1e-5)
  assert ("With your trunk square to your feet, a thing at x in your stance frame is in the "
          f"picture only if it stands lower than ({at_feet:.2f} - {drop:.2f} x) m over the floor") in spot
  assert "a tag reads only with the whole of it inside the picture" in spot
  shoulder = float(data.xanchor[model.joint("left_shoulder_pitch").id][2])
  small = (at_feet - shoulder - tags.BLOCK_TAG_SIZE / 2) / drop
  assert ("the lower a tag is, the further off it reads, a small one level with your shoulders "
          f"only within about {small:.1f} m ahead of you, a larger one nearer still, and nothing "
          "as high as your camera ever") in spot
  assert f"a small block's tag reads from up to about {st.SPOT_BLOCK_READS_M:g} m away" in spot
  watch = " ".join(st.HANDS_VERBS["watch"].doc.split())
  assert "your colour camera" in watch and "head camera" not in watch
  assert (f"the top of whose picture is {colour['top']:.0f} degrees below level as you stand "
          "and lower as you bow") in watch
  assert "navigation camera" in " ".join(st.VERBS["find"].doc.split())
  told = " ".join(mp.system_prompt(st.hand_facts()).split())
  for doc in (spot, watch):
    assert doc in told
  assert "head camera" not in told


def test_a_wall_tag_is_above_spots_picture_from_the_counter_and_in_the_navigation_cameras():
  """#660's premise, one `mj_forward` and no flight. With the camera right
  over the counter's front edge -- no stance gets it nearer the wall -- a
  tag on the wall above the counter is above the colour camera's picture
  (`spot`'s, its text says) and inside the navigation camera's (`find`'s and
  `look`'s). And the wall's own tags, 120 mm at the shoulders' height, come
  whole into the colour camera's picture only with it within 0.2 m of them:
  flown, 11-13 cm off a bare wall read them and 16 cm did not."""
  model, data = _standing()
  colour = _picture(model, data, "color_eye")
  navigation = _picture(model, data, "nav_eye")
  for pic in (colour, navigation):
    camera_x, camera_y = float(pic["at"][0]), float(pic["at"][1])
    tag = (camera_x + sr.COUNTER_D, camera_y, sr.TAG_Z)
    assert _in_picture(pic, tag) is (pic is navigation), pic["top"]
  # the whole of a wall tag, whose black square is 120 mm: its top edge is
  # the last to come in (and the picture's own top only falls off with range)
  drop = float(colour["at"][2]) - (sr.TAG_Z + tags.TAG_SIZES[tags.COUNTER_TAG_IDS[1]] / 2)
  in_from = drop / math.tan(math.radians(colour["top"]))
  assert 0.0 < in_from < 0.2, in_from


def test_a_practices_first_picture_is_said_to_be_the_navigation_cameras():
  """#660: the picture a practice opens with (`evaluation.practice.head_picture`,
  `Body.head_camera`) is the navigation camera's, by that name."""
  mind = Mind('def a():\n    wait(1)\n')
  pr = mp.Practice(mind, "a-mind", st.hand_facts(), ep.JOB.format(limit=30), retry_wait_s=0.0)
  pr.first(picture=b"\xff\xd8")
  assert "The picture is your navigation camera's" in json.dumps(mind.asked[0]["messages"])


def test_a_tag_on_every_face_is_read_by_the_face_seen_most_square_on():
  """What `spot`'s and `watch`'s text states (#650): of the faces one
  picture decodes, the one seen most square-on (#407: keyed by id, the last
  decoded won, a knob's top seen edge-on) -- whichever order they decode in."""
  model, data = _standing()
  body = body_for(model, data)

  def face(t, normal):
    return {"id": 54, "t": t, "center": (0.0, 0.0), "yaw": 0.0, "normal": normal,
            "edge": (1.0, 0.0, 0.0)}
  square = face((0.0, 0.0, 0.8), (0.0, 0.0, 1.0))
  glancing = face((0.0, 0.013, 0.813), (0.0, 0.96, 0.28))
  for picture in ([square, glancing], [glancing, square]):
    body._detector = lambda: NS(detect_all=lambda _: list(picture))
    assert body._decoded(54)["range"] == pytest.approx(0.8)
  body._detector = lambda: NS(detect_all=lambda _: [])
  assert body._decoded(54) is None


class CutMind(Mind):
  """A mind whose first answer runs out of its tokens still thinking."""

  def __init__(self, *sources, cut=1):
    super().__init__(*sources)
    self.cut = cut

  def create(self, **kw):
    if self.cut:
      self.cut -= 1
      self.asked.append(kw)
      return NS(content=[NS(type="text", text="")], stop_reason="max_tokens",
                usage=NS(input_tokens=100, output_tokens=kw["max_tokens"]))
    return super().create(**kw)


def test_an_answer_cut_off_is_asked_again_saying_so():
  """37 of #493's answers ran out of their 64,000 tokens still thinking, and
  were asked again as if nothing had happened: the cap is 128,000 now, the
  mind is told its budget, and an answer cut off is asked again saying so --
  once, however often it is cut."""
  mind = CutMind('def a():\n    wait(1)\n', cut=2)
  pr = mp.Practice(mind, "a-mind", st.hand_facts(), ep.JOB.format(limit=30), retry_wait_s=0.0)
  turn = pr.first()
  assert [t.kind for t in pr.turns] == ["first", "retry", "retry"]
  assert pr.turns[0].cut and "cut off" in pr.turns[0].error
  assert turn.procedure is not None and not turn.cut
  assert all(kw["max_tokens"] == mp.MAX_TOKENS == 128000 for kw in mind.asked)
  for kw in mind.asked[1:]:
    asked = kw["messages"][-1]["content"]
    assert asked.startswith(mp.CUT_NOTE.format(tokens=mp.MAX_TOKENS))
    assert asked.count("ran out of its") == 1
  prompt = mp.system_prompt(st.hand_facts())
  assert "128,000 tokens" in prompt and "64,000" in prompt


def test_an_answer_whose_stream_closed_is_asked_again_saying_so():
  """#560's table smoke: three answers, each still thinking when the
  router closed its stream at 20 minutes, read as "not JSON" and were
  asked again as if nothing had happened. A stream closed mid-answer is a
  cut, and the retry says how long and how much it was."""
  class ClosedMind(Mind):
    def __init__(self, *sources):
      super().__init__(*sources)
      self.closed = 1

    def create(self, **kw):
      if self.closed:
        self.closed -= 1
        self.asked.append(kw)
        return NS(content=[NS(type="text", text="")], stop_reason="closed",
                  usage=NS(input_tokens=100, output_tokens=98703))
      return super().create(**kw)
  mind = ClosedMind('def a():\n    wait(1)\n')
  pr = mp.Practice(mind, "a-mind", st.hand_facts(), ep.JOB.format(limit=30), retry_wait_s=0.0)
  turn = pr.first()
  assert [t.kind for t in pr.turns] == ["first", "retry"]
  cut = pr.turns[0]
  assert cut.cut and cut.closed and cut.error.startswith("cut off: its stream closed after")
  assert cut.as_dict()["closed"] is True and turn.procedure is not None
  asked = mind.asked[1]["messages"][-1]["content"]
  assert asked.startswith(mp.CLOSED_NOTE.format(minutes=cut.wall_s / 60, tokens=98703))
  assert "98,703 tokens" in asked


def test_a_first_answer_lost_to_every_cut_leaves_the_job_ahead_of_the_next_ask():
  """⚠ #560's table arms: the deployed mind's first answer ran out of its
  128,000 tokens and both retries' streams closed at 20 minutes, and the job
  went with them -- a lost reply took what it answered out of the
  conversation -- so every later try was written for a job the mind was
  never shown (tags 1 and 2, a plate slid to "the place"). A lost reply
  loses only itself: what it answered goes ahead of the next ask, said once."""
  mind = CutMind('def a():\n    wait(1)\n', 'def b():\n    wait(2)\n', cut=3)
  job = ep.JOB.format(limit=30)
  pr = mp.Practice(mind, "a-mind", st.hand_facts(), job, retry_wait_s=0.0)
  assert pr.first().answer is None
  assert [t.kind for t in pr.turns] == ["first", "retry", "retry"]
  assert pr.after("TRY 1\n\nYou had no procedure that compiled.").procedure is not None
  sent = mind.asked[-1]["messages"]
  assert [m["role"] for m in sent] == ["user"]
  text = sent[0]["content"]
  assert text.index(job) < text.index(mp.LOST_NOTE) < text.index("TRY 1")
  pr.after("TRY 2\n\nYour procedure ran.")
  sent = mind.asked[-1]["messages"]
  assert [m["role"] for m in sent] == ["user", "assistant", "user"]
  assert json.dumps(sent).count("THE JOB") == 1 and sent[-1]["content"].startswith("TRY 2")
  # ...and a chest's notes, its last ask, are never held: nothing follows them
  mind.cut = 3
  assert pr.carry(None)[1].answer is None and pr._unanswered == []


class _Killed(BaseException):
  """A pool worker killed mid-call (`pool.terminate()`): nothing it was
  doing returns, nor is caught as an error."""


def _answered(source='def a():\n    wait(1)\n'):
  """The router's answer: `source`, 100 tokens in and 40 out."""
  text = json.dumps({"think": "a try", "source": source})
  return 200, {"choices": [{"message": {"content": text}}],
               "usage": {"prompt_tokens": 100, "completion_tokens": 40}}


def _script(monkeypatch):
  """`scripts/practice.py`, loaded."""
  import importlib.util
  from pathlib import Path
  path = Path(__file__).resolve().parents[1] / "scripts" / "practice.py"
  spec = importlib.util.spec_from_file_location("practice_script", path)
  script = importlib.util.module_from_spec(spec)
  monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")     # what its import sets
  spec.loader.exec_module(script)
  return script


def _practice_script(monkeypatch, router, first=()):
  """`scripts/practice.py`, its mind's router `router` (no network, no price
  list) and each set-out flown on the stub world without a picture, in this
  process: the chests in `first` first, and one whose worker was killed
  (`_Killed`) returning nothing."""
  import urllib.request
  from pluggybot import batch
  from pluggybot.mind import llm
  script = _script(monkeypatch)
  monkeypatch.setattr(batch, "watched", lambda: None)   # the runner's, never this process's

  def no_network(*a, **kw):
    raise AssertionError("a test reached the network")
  monkeypatch.setattr(urllib.request, "urlopen", no_network)
  monkeypatch.setattr(llm, "stream_fetch", router)
  monkeypatch.setattr(llm, "_default_fetch", lambda *a, **kw: (503, {}))   # no price list
  monkeypatch.setenv(llm.TOKEN_ENV, "hf_test")
  practice_on = ep.practice_on
  monkeypatch.setattr(ep, "practice_on", lambda k, mind, attempts, **kw: practice_on(
    k, mind, attempts, **{**kw, "picture": False}, build=lambda t: stub_world(t)))

  class Pool:
    def __init__(self, processes, **kw):
      pass

    def __enter__(self):
      return self

    def __exit__(self, *exc):
      return False

    def terminate(self):
      pass

    def imap_unordered(self, fn, jobs):
      for j in sorted(jobs, key=lambda j: j["chest"] not in first):
        try:
          yield fn(j)
        except _Killed:
          pass
  monkeypatch.setattr(script, "get_context", lambda kind: NS(Pool=Pool))
  return script


def test_a_practice_asks_at_the_effort_it_is_given_and_never_pools_two(monkeypatch, tmp_path,
                                                                       capsys):
  """#560's table arms on #626: at its provider's own effort the deployed
  mind thought a median of 95-115k tokens a call, and 97 of 208 answers
  were cut off; the arms fly again asked to think less (Ben, 2026-11-07),
  by its reasoning effort, never a smaller budget. `--effort` goes on every
  request a practice makes, on every row (`run.effort`, the skill), and a
  record read keeps each effort's rows apart; unset, nothing is sent, as
  before. Through the script's `main`, the router faked: no network."""
  bodies = []

  def router(url, body, headers, timeout, **kw):
    bodies.append(body)
    return _answered()
  script = _practice_script(monkeypatch, router)
  low, unset = tmp_path / "low", tmp_path / "unset"
  for into, effort in ((low, ["--effort", "low"]), (unset, [])):
    script.main(["--chests", "0", "--attempts", "1", "--model", "org/m", "--into", str(into),
                 *effort])
  asked = [b.get("reasoning_effort") for b in bodies]
  assert asked == ["low", None] and "reasoning_effort" not in bodies[1]
  rows = {d: [json.loads(line) for line in open(d / "rows.jsonl")] for d in (low, unset)}
  assert [r["run"]["effort"] for d in (low, unset) for r in rows[d]] == ["low", None]
  assert not any("error" in r for d in rows for r in rows[d])
  # read together, the two are two regimes: the per-task record's and the mind's cost
  both = tmp_path / "both"
  both.mkdir()
  (both / "rows.jsonl").write_text("".join(
    json.dumps(r) + "\n" for d in (low, unset) for r in rows[d]))
  capsys.readouterr()
  script.read(str(both))
  said = capsys.readouterr().out
  assert said.count("THE PER-TASK RECORD") == 2 and said.count("THE MIND,") == 2
  assert "org/m at effort low" in said and "org/m at its provider's effort" in said


def test_every_practice_call_is_journaled_as_asked_and_as_answered_the_failed_too(monkeypatch,
                                                                                 tmp_path):
  """#560 (#626's call 6): a set-out's calls are journaled as they go, on
  disk before each goes on -- its ask, then its turn as recorded, a call
  that failed too -- and the journal spends what its rows say it spent."""
  class FailsFirst(Mind):
    def create(self, **kw):
      if not self.asked:
        self.asked.append(kw)
        raise RuntimeError("503 from the router")
      return super().create(**kw)
  script = _script(monkeypatch)
  path = tmp_path / "inflight" / "practice-0.jsonl"
  on_disk = []
  journal = script.journal(path)

  def on_call(turn, answered):
    journal(turn, answered)
    on_disk.append(len(path.read_text().splitlines()))
  pr = mp.Practice(FailsFirst('def a():\n    wait(1)\n', 'def b():\n    wait(2)\n'), "a-mind",
                   st.hand_facts(), ep.JOB.format(limit=30), retry_wait_s=0.0, on_call=on_call)
  rows = ep.practice_on(0, pr, 2, picture=False, budget_s=30.0, build=lambda t: stub_world(t))
  lines = [json.loads(line) for line in path.read_text().splitlines()]
  assert on_disk == [1, 2, 3, 4, 5, 6]
  assert [(ln["kind"], ln["answered"], ln["try"]) for ln in lines] == [
    ("first", False, 1), ("first", True, 1), ("retry", False, 1), ("retry", True, 1),
    ("after", False, 2), ("after", True, 2)]
  assert [ln["tokensIn"] for ln in lines if ln["answered"]] == [0, 100, 100]
  answered = [ln for ln in lines if ln["answered"]]
  assert (sum(ln["tokensIn"] for ln in answered), sum(ln["tokensOut"] for ln in answered),
          len(answered)) == tuple(sum(r["practice"][k] for r in rows)
                                  for k in ("tokensIn", "tokensOut", "calls"))


def test_an_early_stop_writes_what_the_set_outs_it_killed_spent(monkeypatch, tmp_path, capsys):
  """#560 (#626's call 6, Ben, 2026-11-07): the early stop kills the
  set-outs in flight, and their spend reached no row -- #626's table arms
  recorded $8.50 and ~$6.60 more was killed. A killed set-out's spend is
  one record off its journal, apart from the record's rows (a row is an
  attempt): its answered calls summed and priced as a row is, the call it
  had open counted and never priced. A set-out whose rows came has none,
  the per-task record never reads it, and `--read` says it apart. Through
  the script's `main`: chest 1 is in flight, its second call open, when
  chest 0 ends and the bar of 2 is out of reach."""
  from pluggybot.mind import llm
  calls = []

  def router(url, body, headers, timeout, **kw):
    calls.append(body)
    if len(calls) == 2:                  # chest 1's second, still open at the kill
      raise _Killed()
    return _answered()
  script = _practice_script(monkeypatch, router, first=(1,))
  monkeypatch.setattr(llm.HFClient, "pricing", lambda self, model: (0.5, 2.0))
  into = tmp_path / "rec"
  script.main(["--chests", "0-1", "--attempts", "2", "--model", "org/m", "--into", str(into),
               "--early-stop"])
  rows = [json.loads(line) for line in open(into / "rows.jsonl")]
  assert {r["task"]["chest"] for r in rows} == {0} and len(rows) == 2
  assert json.loads((into / "stopped.json").read_text())["after"] == 1
  dead = [json.loads(line) for line in open(into / "killed.jsonl")]
  assert len(dead) == 1
  (gone,) = dead
  assert gone["task"]["chest"] == 1 and gone["killed"] == "stopped early"
  assert gone["process"] == "practice" and gone["skill"] == rows[0]["skill"]
  assert script.regime_of(gone) == script.regime_of(rows[0])
  assert {k: gone["practice"][k] for k in ("tokensIn", "tokensOut", "calls")} == {
    "tokensIn": 100, "tokensOut": 40, "calls": 1}
  assert gone["usd"] == round((100 * 0.5 + 40 * 2.0) / 1e6, 5) == rows[0]["usd"]
  assert [(c["kind"], c["try"]) for c in gone["open"]] == [("after", 2)] and gone["onTry"] == 2
  assert "verdict" not in gone and "attempt" not in gone
  assert not list((into / "inflight").iterdir())
  # read: the killed spend said apart, the per-task record as without it
  capsys.readouterr()
  script.read(str(into))
  said = capsys.readouterr().out
  assert said.count("THE MIND,") == 1
  assert ("  and killed in flight, 1 set-outs (stopped early): 1 answers, 100 tokens in and 40 "
          "out, $0.00, 0.0 h waiting on it; 1 calls cut off uncounted") in said
  script.att.read(str(into))
  record = capsys.readouterr().out
  (into / "killed.jsonl").unlink()
  script.att.read(str(into))
  assert capsys.readouterr().out == record and "THE PER-TASK RECORD" in record


def _pass(name, source, chest, attempt):
  return {"name": name, "source": source, "chest": chest, "attempt": attempt}


def _box(*lines, end="\n"):
  return "def open_box():\n" + "\n".join(f"    {line}" for line in lines) + end


def test_a_robot_keeps_what_passed_the_most_chests_not_its_newest_pass():
  """#535, fixB's practice (#493's stage B): seven passes under one name,
  the first six stepping 6 cm right before the reach, the seventh, on
  chest 14, without the step -- kept as the newest, it was what every
  held-out copy started from, and 0 of 7 passed. Its passes on chests 7 and
  8 were ONE program, the second without the first's last newline: kept by
  the chests passed, that one stays, its step with it."""
  up = 'pull("left", 0, 0, 1, 60, {}, speed=0.05)'
  stepped = ["spot(53)", "step(0, -0.06)", 'reach("left", 0.36, 0.17, 0.82)']
  passes = [_pass("open_box", _box(*stepped, up.format(d)), c, a)
            for d, c, a in ((0.28, 1, 9), (0.03, 3, 2), (0.04, 5, 2))]
  twice = _box(*stepped, up.format(0.05), 'pull("left", -0.5, 0, -0.866, 12, 0.02)')
  passes += [_pass("open_box", twice, 7, 1), _pass("open_box", twice.rstrip("\n"), 8, 1),
             _pass("open_box", _box(*stepped, up.format(0.06)), 10, 5),
             _pass("open_box", _box("spot(53)", 'reach("left", 0.37, 0.16, 0.81)', up.format(0.025)),
                   14, 4)]
  kept, dropped = ep.keep(passes)
  assert kept == [{"name": "open_box", "source": twice.rstrip("\n"), "chest": 8, "attempt": 1,
                   "passed": [[7, 1], [8, 1]]}] and dropped is None
  assert "step(0, -0.06)" in kept[0]["source"]


def test_the_most_passes_wins_under_a_name_the_newest_among_ties_and_the_oldest_name_goes():
  a, b = "def a():\n    wait(1)\n", "def a():\n    wait(2)\n"
  a_again = "def a():\n\n    wait(1)    # once more\n"
  kept, dropped = ep.keep([_pass("a", a, 3, 2)])
  assert kept == [{"name": "a", "source": a, "chest": 3, "attempt": 2, "passed": [[3, 2]]}]
  assert dropped is None
  # a pass each: the newest
  passes = [_pass("a", a, 3, 2), _pass("a", b, 4, 1)]
  assert ep.keep(passes)[0][0]["source"] == b
  # its first program again, laid out and commented otherwise, is one
  # program twice -- counted from before the second took its place -- and
  # carried as its newest text
  passes.append(_pass("a", a_again, 5, 1))
  kept, _ = ep.keep(passes)
  assert kept[0]["source"] == a_again and kept[0]["passed"] == [[3, 2], [5, 1]]
  # ...kept over a newer program passing once
  passes.append(_pass("a", "def a():\n    wait(5)\n", 6, 1))
  assert ep.keep(passes)[0][0]["source"] == a_again
  # ...and a tie of two again goes to the newest
  passes.append(_pass("a", b, 7, 1))
  assert ep.keep(passes)[0][0]["passed"] == [[4, 1], [7, 1]]
  # names in the order they last passed; past the cap the oldest goes
  passes.append(_pass("b", "def b():\n    wait(3)\n", 8, 1))
  kept, dropped = ep.keep(passes)
  assert [k["name"] for k in kept] == ["a", "b"] and dropped is None
  assert [k["name"] for k in ep.keep(passes, cap=3)[0]] == ["a", "b"]
  passes.append(_pass("a", b, 9, 2))
  kept, dropped = ep.keep(passes, cap=2)
  assert [k["name"] for k in kept] == ["b", "a"] and dropped is None
  passes.append(_pass("c", "def c():\n    wait(4)\n", 10, 1))
  kept, dropped = ep.keep(passes, cap=2)
  assert [k["name"] for k in kept] == ["a", "c"] and dropped == "b"
  # one the language cannot read is its own text
  assert ep.program("def (") == "def (" and ep.program(a) == ep.program(a_again)


def test_the_bar_is_out_of_reach_once_more_chests_failed_within_its_tries_than_it_allows():
  """Ben, 2026-10-18: stop once it is plain the bar cannot be met. 80 % of
  32 is 26 within 3: a seventh chest failed puts it out of reach, and a pass
  at the fourth try is a failure within 3."""
  assert not ep.out_of_reach([None] * 6, 32)
  assert ep.out_of_reach([None] * 6 + [4], 32)
  assert not ep.out_of_reach([1, 2, 3] * 8 + [None] * 6, 32)
  assert ep.out_of_reach([None] * 3, 10) and not ep.out_of_reach([None] * 2, 10)


def _chest(first, name="opener", error=False):
  """A chest's rows that first passed at `first` (None: never), admitting
  `name`."""
  if error:
    return [{"error": "it raised"}]
  rows = [{"attempt": n, "verdict": {"ok": n == first}, "practice": {"source": f"def {name}(): ..."}}
          for n in range(1, (first or 5) + 1)]
  if first:
    rows[-1]["admitted"] = name
  return rows


def test_the_practice_first_carries_only_what_passed_and_stops_once_it_holds():
  outcomes = {0: 4, 1: 1, 2: None, 3: 1, 4: 1, 5: 1, 6: 1}
  seen = []

  def run_chest(k, kept, carried):
    seen.append((k, [c["name"] for c in kept]))
    return _chest(outcomes[k], name=f"p{k}")
  kept, how = ep.acquire(run_chest, range(7), until=3)
  # it carried chest 0's pass to chest 1, and stopped at three running passes
  assert seen[:2] == [(0, []), (1, ["p0"])] and len(seen) == 6
  assert how["firsts"][2] == [2, None] and "3 chests running" in how["ended"]
  assert [c["name"] for c in kept] == ["p0", "p1", "p3", "p4", "p5"]
  # ...and gives up with nothing kept, flying no more
  kept, how = ep.acquire(lambda k, kept, carried: _chest(None), range(20), give_up_after=4)
  assert kept == [] and len(how["firsts"]) == 4 and "given up" in how["ended"]
  # ...and a chest whose practice raised is no pass
  kept, how = ep.acquire(lambda k, kept, carried: _chest(1, error=k == 1), range(4), until=3)
  assert how["firsts"][1] == [1, None] and len(how["firsts"]) == 4


def test_the_flight_stops_early_once_the_bar_is_out_of_reach_and_not_without_asking():
  results = [_chest(None)] * 10
  written = []
  stopped = ep.flown(iter(results), 10, True, written.append)
  assert stopped["after"] == 3 and stopped["failedWithin"] == 3 and len(written) == 3
  assert ep.flown(iter(results), 10, False) is None


def test_a_chest_is_shown_what_the_robot_kept():
  kept = [{"name": "opener", "source": "def opener():\n    wait(1)\n", "chest": 7, "attempt": 2}]
  mind = Mind('def b():\n    wait(2)\n')
  pr = mp.Practice(mind, "a-mind", st.hand_facts(), ep.JOB.format(limit=30))
  rows = ep.practice_on(0, pr, 1, kept=kept, picture=False, budget_s=30.0,
                        build=lambda t: stub_world(t))
  first = mind.asked[0]["messages"][0]["content"]
  assert first.startswith("YOUR LIBRARY") and "opener -- passed box 7 on its try 2" in first
  assert "def opener():\n    wait(1)" in first and "THE JOB" in first
  assert rows[0]["kept"] == ["opener"] and rows[0]["practice"]["source"].startswith("def b")


def test_a_hand_reaches_its_own_side_as_the_body_plans_it_nothing_moved():
  """What a reach out of reach says of the other hand is the reach's own
  plan, in the believed world frame (`Body.hand_reaches`, `Reach.plans`):
  the humanoid at its counter, the left hand asked a point 0.5 m to its
  side, which the right reaches no longer -- turned at the waist, it plans
  to 0.4 m past the middle (#560's stage (c3)) -- and nothing moved by
  asking."""
  from pluggybot.evaluation import attempts as at
  world = at.set_out(at.Task(chest=0, body="humanoid", mount="counter"))
  try:
    body = world.body
    left = body.to_world(np.array([0.35, 0.50, 0.85]))
    before = body.reach.phase["left"], tuple(body.reach.posture)
    assert body.hand_reaches("left", left)
    assert not body.hand_reaches("right", left)
    assert (body.reach.phase["left"], tuple(body.reach.posture)) == before
  finally:
    world.close()


# ---- by feel and by sight: what a hand felt, and the depth camera's flat top -----


def test_a_hand_verb_says_what_it_felt_and_the_report_shows_it():
  """The robot's own senses of the hand while each of its verbs ran -- its
  grip's place, the force, whether it holds -- first and last kept: told
  only how a verb ended, a mind shut its jaws on nothing three tries
  running (#493)."""
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n    close_hand("left")\n')
  reach, close = ran["steps"]
  for step in (reach, close):
    assert step["felt"][0]["t"] == 0.0 and len(step["felt"]) >= 2
    assert set(step["felt"][0]) == {"t", "at", "force", "held"}
  assert reach["felt"][-1]["at"] == pytest.approx([0.3, 0.2, 0.9])
  assert not close["felt"][0]["held"] and close["felt"][-1]["held"]
  text = mp.report_text(1, ran, {"ok": False, "reason": "no"}, 1.0, False, 1)
  assert "felt -- s: grip x y z, m | force it pushed with x y z, N | holding" in text
  assert "0.300 0.200 0.900 | +0.0 +0.0 +0.0 | yes" in text
  assert '"felt"' not in text


def test_felt_is_thinned_evenly_keeping_its_ends():
  felt = [{"t": i} for i in range(20)]
  kept = st._thinned(felt)
  assert len(kept) == st.FELT_MAX and kept[0]["t"] == 0 and kept[-1]["t"] == 19
  assert st._thinned(felt[:5]) == felt[:5]


def test_what_is_felt_drives_its_routine_as_yield_from_would():
  """A routine run under `_felt_routine` is sent, thrown into and closed as `yield
  from` would: what the driver throws (a hook's `MissionAborted`) reaches
  it, and its `finally` runs."""
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
  body = _stub()
  life = _Life(body)
  gen = st._felt_routine(life, "left", routine())
  assert next(gen) == "a" and gen.send(None) == "b"
  assert gen.throw(KeyError("x")) == "caught"
  with pytest.raises(StopIteration) as stop:
    gen.send(None)
  assert stop.value.value[0] == "done" and ended == [True]
  # ...closed, it closes the routine: held here, nothing else would
  inner = routine()
  gen = st._felt_routine(life, "left", inner)
  next(gen)
  gen.close()
  assert ended == [True, True]


#: Where the head's D435 is in the stance frame at the stand, m, and how far
#: it looks down, rad: its `depth_eye` on the robot's model of itself
#: (`arms.SelfModel` posed at `stand_qpos`, 2026-11-07), and its height's
#: vertical field, rad.
D435_AT, D435_PITCH = np.array([0.045, 0.0, 1.26]), math.radians(40.0)
D435_FOVY = math.radians(58.0)


@functools.lru_cache(maxsize=16)
def _stage(boxes: tuple) -> dict:
  """`boxes` and the floor as the D435 sees them, standing, before its
  noise: each `(x0, x1, y0, y1, z0, z1)` in the stance frame. Each pixel's
  ray cast at them (the nearest, as the part sees), the same range gate and
  the same shadow as `DepthCamera`'s."""
  from pluggybot.perception import box, depth
  w, h = box.DEPTH_W, box.DEPTH_H
  focal = (h / 2) / math.tan(D435_FOVY / 2)
  c, s = math.cos(D435_PITCH), math.sin(D435_PITCH)
  # camera to stance frame: its x (right), y (up) and z (back, out of its lens)
  turn = np.column_stack([[0.0, -1.0, 0.0], [s, 0.0, c], [-c, 0.0, s]])
  u, v = np.meshgrid(np.arange(w) - w / 2 + 0.5, np.arange(h) - h / 2 + 0.5)
  rays = np.stack([u / focal, -v / focal, -np.ones_like(u)], axis=-1) @ turn.T
  eye = D435_AT
  with np.errstate(divide="ignore", invalid="ignore"):
    z = np.where(rays[..., 2] < 0, -eye[2] / rays[..., 2], np.inf)
    for (x0, x1, y0, y1, z0, z1) in boxes:
      lo, hi = (np.array([x0, y0, z0]) - eye) / rays, (np.array([x1, y1, z1]) - eye) / rays
      t0, t1 = np.minimum(lo, hi).max(axis=-1), np.maximum(lo, hi).min(axis=-1)
      z = np.where((t1 >= t0) & (t0 > 0.0) & (t0 < z), t0, z)
  z = np.where(np.isinf(z) | (z > depth.MAX_Z), np.nan, z)
  flat = z.reshape(-1)
  shadow = depth.DepthCamera._shadow(
    NS(focal_px=focal, baseline=depth.BASELINE, _u=u.reshape(-1), height=h, width=w),
    flat).reshape(z.shape)
  return {"z": z, "shadow": shadow, "rays": rays, "eye": eye, "turn": turn, "focal": focal}


def _frame(stage: dict, pose, rng):
  """One frame of the stage through the part's noise and dropout, laid in
  the believed world frame the stub stands at (`DepthPicture`)."""
  from pluggybot.perception import box, depth
  z = np.where(stage["shadow"] | (stage["z"] < box.DEPTH_MIN_Z), np.nan, stage["z"])
  z = z + rng.normal(0.0, 1.0, z.shape) * depth.NOISE_K * np.nan_to_num(z) ** 2
  z = np.where(rng.random(z.shape) < depth.DROPOUT, np.nan, z)
  valid = ~np.isnan(z)
  points = stage["eye"] + stage["rays"][valid] * z[valid][:, None]
  x, y, yaw = pose
  yaw_turn = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0],
                       [0.0, 0.0, 1.0]])
  return depth.DepthPicture(z=z, points=points @ yaw_turn.T + [x, y, 0.0],
                            eye=yaw_turn @ stage["eye"] + [x, y, 0.0],
                            turn=yaw_turn @ stage["turn"], focal=stage["focal"],
                            near=box.DEPTH_MIN_Z, reach=depth.MAX_Z, noise_k=depth.NOISE_K)


class Sees(StubBody):
  """A stub whose D435 sees `boxes` and the floor (`_stage`), laid in the
  stance frame it stands in."""
  boxes = ()

  def depth_pictures(self, width, height, min_z, frames=1, apart=0.1):
    yield from self._wait(frames * apart)
    rng = np.random.default_rng(636)
    return [_frame(_stage(tuple(self.boxes)), self.pose, rng) for _ in range(int(frames))]


def _sees(boxes=(), pose=(0.0, 0.0, 0.0)) -> Sees:
  body = Sees(*_bare())
  body.boxes = boxes
  body.start_at(*pose)
  return body


#: A counter's top and, on it, a box's flat top 0.25 m deep and 0.30 m wide.
COUNTER = (0.3, 1.2, -0.6, 0.6, 0.0, 0.72)
BOX_ON_IT = (0.40, 0.65, 0.02, 0.32, 0.72, 0.87)


def test_measure_reads_the_flat_top_at_a_point_and_its_edges_in_the_stance_frame():
  """Sight beside feel (#493, its own arm): the highest flat face round the
  point named -- never the wider counter under it -- and its four edges,
  wherever the body stands and faces."""
  for pose in ((0.0, 0.0, 0.0), (2.0, 1.0, math.pi / 2)):
    ran = _fly('def a():\n    measure(0.36, 0.2)\n    t = read("object.top")\n'
               '    n = read("object.near")\n    f = read("object.far")\n'
               '    l = read("object.left")\n    r = read("object.right")\n',
               _sees([COUNTER, BOX_ON_IT], pose), measure=True)
    got = ran["locals"]
    assert got["t"] == pytest.approx(0.87, abs=0.005)
    assert (got["n"], got["f"]) == pytest.approx((0.40, 0.65), abs=0.01)
    assert (got["l"], got["r"]) == pytest.approx((0.32, 0.02), abs=0.01)
  ran = _fly('def a():\n    measure(2.5, 1.5)\n', _sees([COUNTER, BOX_ON_IT]), measure=True)
  assert ran["steps"][-1]["reason"] == "no flat top seen round (2.5, 1.5) from where you stand"
  # ...and a body with no camera looks at nothing, no picture to read past
  ran = _fly('def a():\n    measure(0.5, 0)\n', _stub(), measure=True)
  assert ran["steps"][-1]["reason"] == "no flat top seen round (0.5, 0) from where you stand"
  assert "measure" not in st.hand_facts(measure=False).verbs
  assert "object.top" not in st.hand_facts(measure=False).sensors
  assert "object.near_taller" not in st.hand_facts(measure=False).sensors


def test_measure_never_reads_its_windows_rim_as_where_a_top_ends():
  """#630: `measure` looks within `MEASURE_NEAR_M` of its point, and a top
  that runs on past that ends, in its points, at the window's rim -- from
  (0.9, 0) Luca's counter read `object.near` 0.453-0.456 however it
  stepped, and it crept on that toward the counter. An end that is the
  window's is no reading, and reading it says why; an end inside the
  window still reads, wherever the robot stands."""
  for pose in ((0.0, 0.0, 0.0), (2.0, 1.0, math.pi / 2)):
    ran = _fly('def a():\n    measure(0.9, 0)\n    t = read("object.top")\n'
               '    f = read("object.far")\n    n = read("object.near")\n',
               _sees([COUNTER], pose))
    got = ran["locals"]
    assert "n" not in got, f"the rim, read as where the top begins: {got.get('n')}"
    assert ran["steps"][-1]["reason"] == (
      "object.near has no reading: your last `measure` looked within 0.45 m of (0.9, 0), "
      "and the top it saw runs on past that toward you")
    assert (got["t"], got["f"]) == pytest.approx((0.72, 1.2), abs=0.01)
    assert ran["steps"][0]["ok"] and ran["steps"][0]["runsOn"] == ["near", "left", "right"]
  # ...measured nearer, its near end is in the window, and its far end is not
  ran = _fly('def a():\n    measure(0.5, 0)\n    n = read("object.near")\n'
             '    f = read("object.far")\n', _sees([COUNTER]))
  assert ran["locals"]["n"] == pytest.approx(0.3, abs=0.01)
  assert ran["steps"][-1]["reason"].endswith("runs on past that away from you")


def test_an_end_whose_tops_own_pixels_go_on_past_it_is_no_end(monkeypatch):
  """#636: walking out from an end's points, the picture shows more of the
  top past where the end reads (at the chest, the counter going on beside
  it, read a few cm short at 9 of 2,399 ends past the rim's check): the
  camera never saw it stop, so it is no end, and reading it says so.
  `box.past_end` tells it apart (`test_box.py`); here, the verb's branch."""
  from pluggybot.perception import box
  monkeypatch.setattr(box, "past_end", lambda *a, **k: {
    "lower": 0.0, "taller": 0.0, "cut": 0.0, "runs": 1.0, "reach": False, "near": False})
  ran = _fly('def a():\n    measure(0.55, 0)\n    n = read("object.near")\n', _sees([COUNTER]))
  assert "n" not in ran["locals"]
  assert ran["steps"][-1]["reason"] == ("object.near has no reading: your last `measure` saw the "
                                        "top run on past where it read its end toward you")
  assert "near" in ran["steps"][0]["runsOn"]


def test_measure_reads_no_end_its_picture_cut_and_one_it_saw_past():
  """#636: standing where it works at a counter whose front is 0.22-0.24 m
  ahead, the D435's picture meets the counter's height 0.26 m ahead, so the
  counter's points end at the picture's bottom edge: `object.near` read
  0.26 wherever it looked and however near it stood. The end the picture
  cut is no reading, and reading it says why; a top whose near face the
  camera sees reads it."""
  cut_off = (0.22, 1.2, -0.6, 0.6, 0.0, 0.72)
  source = ('def a():\n    measure(0.55, 0)\n    t = read("object.top")\n'
            '    x = read("object.near_taller")\n')
  for pose in ((0.0, 0.0, 0.0), (2.0, 1.0, math.pi / 2)):
    ran = _fly(source.replace("near_taller", "near"), _sees([cut_off], pose))
    assert ran["locals"]["t"] == pytest.approx(0.72, abs=0.005) and "x" not in ran["locals"]
    assert ran["steps"][-1]["reason"] == (
      "object.near has no reading: your last `measure` saw the top run off your depth "
      "camera's picture toward you")
    assert ran["steps"][0]["ok"] and ran["steps"][0]["unseen"] == ["near"]
    # ...its `_taller` reading says the same, for a read with no number has no other word
    ran = _fly(source, _sees([cut_off], pose))
    assert ran["steps"][-1]["reason"].startswith("object.near_taller has no reading: "
                                                 "your last `measure` saw the top run off")
  # ...and a top whose near face it sees reads it, nothing taller standing past it
  ran = _fly('def a():\n    measure(0.55, 0)\n    n = read("object.near")\n'
             '    x = read("object.near_taller")\n', _sees([COUNTER]))
  assert ran["locals"]["n"] == pytest.approx(0.3, abs=0.01) and ran["locals"]["x"] == 0
  assert "unseen" not in ran["steps"][0] and "taller" not in ran["steps"][0]


@pytest.mark.parametrize("saw, reads, taller", [
  ({"lower": 0.66, "cut": 0.34}, True, False),   # the counter's left end at the picture's corner
  ({"lower": 0.09, "taller": 0.465, "runs": 0.445}, True, True),   # beside the chest
  ({"cut": 0.6, "lower": 0.4}, False, False),
  ({"runs": 0.4, "lower": 0.35, "taller": 0.25}, False, False),
  ({"lower": 0.5, "cut": 0.5}, False, False),    # a tie is the cautious word's
  ({"lower": 0.5, "runs": 0.5}, False, False),
  ({"lower": 0.5, "taller": 0.5}, True, True),
])
def test_an_end_takes_the_walks_most_common_word(monkeypatch, saw, reads, taller):
  """#636: a top's end can be seen past along part of it and cut along the
  rest: what it reads is what most of its walks said, and a tie is the word
  that claims less (`steps.MEASURE_WORDS`) -- a fixed share let a right end
  read as one with nine in a hundred walks saying the top stopped there."""
  from pluggybot.perception import box
  monkeypatch.setattr(box, "past_end", lambda *a, **k: {
    "lower": 0.0, "taller": 0.0, "cut": 0.0, "runs": 0.0, "reach": False, "near": False, **saw})
  ran = _fly('def a():\n    measure(0.55, 0)\n    n = read("object.near")\n'
             '    x = read("object.near_taller")\n', _sees([COUNTER]))
  assert ("n" in ran["locals"]) is reads
  assert (ran["locals"].get("x", 0.0) == 1.0) is taller


def test_a_taller_thing_on_a_top_is_where_its_free_surface_ends_and_the_read_says_so():
  """#636, Ben's call: a taller thing standing on a top is the end of the
  top's free surface: it reads as that end -- its near edge where the top
  runs on behind it -- and the read says that is what it is, `object.*_taller`
  and the step's verdict both."""
  counter = (0.3, 1.0, -0.6, 0.6, 0.0, 0.72)
  against = (0.3, 1.0, -0.6, -0.1, 0.72, 1.05)
  source = ('def a():\n    measure(0.65, 0.3)\n    t = read("object.top")\n'
            '    r = read("object.right")\n    x = read("object.right_taller")\n'
            '    l = read("object.left_taller")\n')
  ran = _fly(source, _sees([counter, against]))
  assert ran["locals"]["t"] == pytest.approx(0.72, abs=0.005)
  assert ran["locals"]["r"] == pytest.approx(-0.1, abs=0.01)
  assert (ran["locals"]["x"], ran["locals"]["l"]) == (1.0, 0.0)
  assert ran["steps"][0]["taller"] == ["right"]
  # ...a taller thing that hides the top behind it: the end is where the camera lost it
  behind = (0.42, 0.72, -0.25, 0.1, 0.72, 1.05)
  ran = _fly('def a():\n    measure(0.75, 0.5)\n    r = read("object.right")\n'
             '    x = read("object.right_taller")\n', _sees([counter, behind]))
  assert ran["locals"]["x"] == 1.0 and 0.09 <= ran["locals"]["r"] <= 0.16
  assert ran["steps"][0]["taller"] == ["right"]
  # ...and no taller thing, no taller reading: the same top, its end seen past
  ran = _fly(source, _sees([(0.3, 1.0, -0.1, 0.6, 0.0, 0.72)]))
  assert ran["locals"]["r"] == pytest.approx(-0.1, abs=0.01)
  assert (ran["locals"]["x"], ran["locals"]["l"]) == (0.0, 0.0)
  assert "taller" not in ran["steps"][0]


def test_measure_looks_past_its_cameras_shadow_and_past_the_floor_it_cannot_reach():
  """#636: the D435's shadow is a band of nothing beside a near thing's
  image-left, tens of pixels wide: walked past, what stands behind it says
  the top ended there -- and a table's far edge across a room, the floor
  beyond it past the part's 3 m, reads where it does. Neither is unseen."""
  # a top standing left of the robot's middle: its left end's shadow, on the
  # floor beside it, is in the picture and the floor beyond it reads lower
  ran = _fly('def a():\n    measure(0.75, 0.25)\n    l = read("object.left")\n',
             _sees([(0.3, 1.2, -0.6, 0.55, 0.0, 0.72)]))
  assert ran["locals"]["l"] == pytest.approx(0.55, abs=0.015)
  assert "unseen" not in ran["steps"][0]
  # ...a table across the room: its far edge at 2.3 m, the floor beyond 3 m
  far_table = (1.6, 2.3, -0.5, 0.5, 0.0, 0.72)
  ran = _fly('def a():\n    measure(2.0, 0)\n    f = read("object.far")\n',
             _sees([far_table]))
  assert ran["locals"]["f"] == pytest.approx(2.3, abs=0.03)
  assert "unseen" not in ran["steps"][0]


def test_a_top_past_the_cameras_reach_has_no_end_that_way():
  """#636: a top's points also end at the part's reach, 3 m, and walked past
  there is nothing to see: an end whose points stand there is the camera's,
  and reading it says so."""
  long_low = (1.2, 6.0, -0.5, 0.5, 0.0, 0.3)
  ran = _fly('def a():\n    measure(2.8, 0)\n    f = read("object.far")\n', _sees([long_low]))
  assert "f" not in ran["locals"]
  assert ran["steps"][-1]["reason"] == (
    "object.far has no reading: your last `measure` saw the top run on past your depth "
    "camera's 3 m reach away from you")
  assert ran["steps"][0]["unseen"] == ["far"]


def test_a_top_nearer_than_the_camera_reads_has_no_end_that_way():
  """#636's review: a high top close under the camera ends, in its points,
  where the part's nearest reading does -- nearer, every pixel reads
  nothing, and a walk across them would say the top stopped there."""
  shelf = (0.0, 0.7, -0.5, 0.5, 0.0, 1.16)
  ran = _fly('def a():\n    measure(0.4, 0)\n    t = read("object.top")\n'
             '    n = read("object.near")\n', _sees([shelf]))
  assert ran["locals"]["t"] == pytest.approx(1.16, abs=0.005) and "n" not in ran["locals"]
  assert ran["steps"][-1]["reason"] == (
    "object.near has no reading: your last `measure` saw the top run on nearer than your depth "
    "camera reads, 0.17 m, toward you")
  assert "near" in ran["steps"][0]["unseen"]


def test_a_lower_surface_just_past_a_far_end_is_not_the_top_running_on():
  """#636's review: seen from far off, a surface 1.7 cm under a table's top
  and just past its far end lies inside the noise of a ray, 3 cm at 1.8 m,
  and passed for the top going on. But the ray's noise is along the ray: in
  height it is 3 mm, and the end stands."""
  ran = _fly('def a():\n    measure(1.6, 0)\n    f = read("object.far")\n'
             '    x = read("object.far_taller")\n',
             _sees([(1.3, 1.85, -0.5, 0.5, 0.0, 0.72), (1.9, 2.4, -0.6, 0.6, 0.0, 0.703)]))
  assert ran["locals"]["f"] == pytest.approx(1.85, abs=0.03), ran["steps"][-1]
  assert ran["locals"]["x"] == 0.0 and "far" not in ran["steps"][0].get("runsOn", ())


def test_measure_sees_no_top_in_the_floor_its_camera_lifts():
  """#630: the D435's noise lifts the floor's own points -- 6-9 mm at 1.3-
  1.6 m -- and a window of bare floor read as a flat top 1.2-2.6 cm up
  (Rowan's chest_diag: 0.0168 m, "garbage"). Under `box.FLOOR_M` is the
  floor's, and a real top over it still reads."""
  from pluggybot.perception import box
  for x, y in ((1.2, -0.9), (2.0, -1.5), (2.6, 0.0)):
    source = (f'def a():\n    k = 0\n    if measure({x:g}, {y:g}):\n        k = 1\n'
              '    t = read("object.top")\n')
    ran = _fly(source, _sees())
    assert not ran["steps"][0]["ok"], f"the floor read as a top at ({x}, {y})"
    assert ran["steps"][-1]["reason"] == (
      f"object.top has no reading: your last `measure` saw no flat top round ({x:g}, {y:g}) "
      "from where you stood")
    # ...and a box's top 0.10 m up still reads
    lid = (x - 0.15, x + 0.15, y - 0.15, y + 0.15, 0.0, 0.10)
    ran = _fly(source, _sees([lid]))
    assert ran["steps"][0]["ok"] and ran["locals"]["t"] > box.FLOOR_M
    assert ran["locals"]["t"] == pytest.approx(0.10, abs=0.02)


def test_measures_text_says_where_it_looks_and_what_is_the_floor():
  """#630: the window was in no text, so the rim read as a top's end and a
  robot could not tell. Its words are read off the constants."""
  from pluggybot.perception import box
  doc = st.HANDS_VERBS["measure"].doc
  assert f"within {st.MEASURE_NEAR_M:g} m of (x, y)" in doc
  assert f"past those {st.MEASURE_NEAR_M:g} m, off your camera's picture" in doc
  assert f"its range ({box.DEPTH_MIN_Z:g} to {st.MEASURE_REACH_M:g} m)" in doc
  assert "its end that way has no reading, and reading it says why" in doc
  assert "reads where the free top stops" in doc and "`object.near_taller`" in doc
  assert f"{box.FLOOR_M:g} m or more over the floor" in doc
  # ...and an `object.*` read before any `measure` names the sensor read
  ran = _fly('def a():\n    wait(0)\n    t = read("object.top")\n')
  assert ran["steps"][-1]["reason"] == "object.top has no reading: `measure` has not run"


def test_an_arm_without_step_has_no_step_to_call_read_or_be_told_of():
  """#493's stage B flies the hands' vocabulary without `step` (Ben,
  2026-10-21): told neither hand reached, 5 robots in 7 that had it walked
  away from a knob their left hand could reach. Gone, it is gone from the
  language, the readings and every word of the prompt."""
  facts = st.hand_facts(step=False)
  assert "step" not in facts.verbs and "reach" in facts.verbs
  assert not [s for s in facts.sensors if s.startswith("step.")]
  text = mp.system_prompt(facts)
  assert "step(" not in text and "step.dx" not in text and "`step`" not in text
  with pytest.raises(st.Refused):
    lang.compile_procedure('def a():\n    step(0.1)\n', facts)
  with pytest.raises(st.Refused):
    lang.compile_procedure('def a():\n    x = read("step.dx")\n', facts)
  assert "step" in st.hand_facts().verbs and "step.dx" in st.hand_facts().sensors


def test_a_world_that_names_no_verbs_is_told_its_vocabulary_and_no_step():
  """#669's review: the body asked `"step" in facts.verbs`, and facts that
  name no verbs -- None, every one of `VERBS` as `language` and
  `describe_vocabulary` read it, the walks and no hand's -- raised
  TypeError. Its vocabulary has no `step`, so it is told none: the walk
  sentence goes where the verbs it is told include `step`."""
  from dataclasses import replace
  facts = replace(st.hand_facts(step=False), verbs=None)
  text = mp.system_prompt(facts)
  assert "step" not in st.language(facts) and "drive_to(" in text and "move(" in text
  assert "step(" not in text and "`step`" not in text


class NotesMind(Mind):
  """A mind that answers what it is asked for: a procedure, or what it
  carries on (`notes_schema`)."""

  def __init__(self, *sources, notes="use the left hand", draft="def d():\n    wait(1)\n"):
    super().__init__(*sources)
    self.notes = list(notes) if isinstance(notes, list) else [notes]
    self.draft = draft

  def create(self, **kw):
    if "notes" in kw["output_config"]["format"]["schema"]["properties"]:
      self.asked.append(kw)
      notes = self.notes.pop(0) if len(self.notes) > 1 else self.notes[0]
      text = json.dumps({"think": "done", "notes": notes, "draft": self.draft})
      return NS(content=[NS(type="text", text=text)], usage=NS(input_tokens=50, output_tokens=20))
    return super().create(**kw)


def test_what_a_chest_ends_with_is_carried_and_shown_at_the_next():
  """Ben, 2026-10-18: what a person takes from a failed attempt. The mind
  writes its notes and a draft as a chest ends -- pass or fail -- its
  answer's cost counted on the chest's last row; the next chest opens
  with them."""
  mind = NotesMind('def a():\n    wait(1)\n')
  pr = mp.Practice(mind, "a-mind", st.hand_facts(), ep.JOB.format(limit=30))
  rows = ep.practice_on(0, pr, 1, write_notes=True, picture=False, budget_s=30.0,
                        build=lambda t: stub_world(t))
  assert rows[-1]["carried"] == {"notes": "use the left hand", "draft": "def d():\n    wait(1)"}
  asked = mind.asked[-1]["messages"][-1]["content"]
  assert asked.startswith("THIS BOX IS DONE: none of your tries passed")
  assert rows[-1]["practice"]["calls"] == 2 and rows[-1]["tokens"] == 140 + 70
  # ...shown at the next chest's start, its own words
  mind2 = NotesMind('def b():\n    wait(1)\n')
  pr2 = mp.Practice(mind2, "a-mind", st.hand_facts(), ep.JOB.format(limit=30))
  rows2 = ep.practice_on(1, pr2, 1, carried=rows[-1]["carried"], picture=False, budget_s=30.0,
                         build=lambda t: stub_world(t))
  first = mind2.asked[0]["messages"][0]["content"]
  assert first.startswith("YOUR NOTES, written after your last box:\n\nuse the left hand")
  assert "YOUR DRAFT" in first and "def d():" in first and rows2[0]["carriedIn"]
  assert "carried" not in rows2[-1]
  # ...notes past the cap are asked for again, shorter, once -- cut, a smoke
  # run's lost what it meant to try next
  over = "x" * (mp.NOTES_CHARS + 50)
  again = NotesMind('def a():\n    wait(1)\n', notes=[over, "shorter"])
  pr3 = mp.Practice(again, "a-mind", st.hand_facts(), ep.JOB.format(limit=30))
  rows3 = ep.practice_on(0, pr3, 1, write_notes=True, picture=False, budget_s=30.0,
                         build=lambda t: stub_world(t))
  assert rows3[-1]["carried"]["notes"] == "shorter"
  assert again.asked[-1]["messages"][-1]["content"].startswith(
    f"Your notes are {len(over)} characters, past the {mp.NOTES_CHARS}")
  # ...and still past it, cut, and said so
  long = NotesMind('def a():\n    wait(1)\n', notes=over)
  pr4 = mp.Practice(long, "a-mind", st.hand_facts(), ep.JOB.format(limit=30))
  rows4 = ep.practice_on(0, pr4, 1, write_notes=True, picture=False, budget_s=30.0,
                         build=lambda t: stub_world(t))
  assert rows4[-1]["carried"]["notes"].endswith(f"[cut at {mp.NOTES_CHARS} characters]")


def test_the_practice_first_carries_the_notes_from_chest_to_chest():
  got = []

  def run_chest(k, kept, carried):
    got.append(dict(carried))
    rows = _chest(None)
    rows[-1]["carried"] = {"notes": f"after {k}", "draft": ""}
    return rows
  _, how = ep.acquire(run_chest, range(3), give_up_after=3)
  assert [g["notes"] for g in got] == ["", "after 0", "after 1"]
  assert how["carried"]["notes"] == "after 2"


# ---- the door, learned with no new help (#535) -------------------------------------


def stub_door_world(task):
  """A door's set-out on the stub body (`attempts.set_out`'s shape): the
  cupboard on a bare floor, its latch sensing every step."""
  from pluggybot.activity import door as dr
  from pluggybot.evaluation import attempts as at
  from test_lid import BARE
  spec = mujoco.MjSpec.from_string(BARE)
  dr.attach_cupboard(spec, dr.draw(task.chest), (0.0, 0.0), tags=False)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  body = StubBody(model, data)
  door = dr.Door(model, data, dr.draw(task.chest))
  body.step_hooks.append(lambda: door.sense(model, data))
  return at.World(task=task, model=model, data=data, body=body, chest=None,
                  start=(0.0, 0.0, 0.0), activity=door)


#: The lid's practice texts as #493 flew them: the door's practice may change
#: the job and the thing's name, and nothing of the lid's (#535).
LID_NOTES_ASK = """THIS BOX IS DONE: {how}. Your next box is another like it, and
you will start it as you started this one, with nothing of this
conversation but what you carry. Write what you will carry: "notes" -- what
worked, what did not, what you would try next, at most {cap} characters,
keeping what still holds from any notes you were given -- and "draft": the
procedure you would start the next box from, the whole of it, or "" for
none. A procedure that passed is in your library already."""
LID_SHOWN = ("YOUR LIBRARY\n\nProcedures you kept, each because the world graded a try of it a "
             "pass on a box like this one:\n\nopener -- passed box 7 on its try 2:\n"
             "def opener():\n    wait(1)\n\nYOUR NOTES, written after your last box:\n\nn\n\n\n"
             "YOUR DRAFT, the procedure you meant to start this box from (it never passed):\n\n"
             "def d():\n    wait(1)\n\nTHE JOB\n\n")


def _asked(mind) -> str:
  return "\n".join(m["content"] for kw in mind.asked for m in kw["messages"]
                   if m["role"] == "user" and isinstance(m["content"], str))


def test_the_door_is_practiced_on_its_own_job_and_nothing_else_changes():
  """#535's stage 2, the door learned with no new help: its practice is the
  lid's but for the job -- the door's text, its time limit and its grader
  -- and what the texts call the thing the robot stands at; the lid's
  texts are as #493 flew them, byte for byte."""
  from pluggybot.evaluation import attempts as at
  kept = [{"name": "opener", "source": "def opener():\n    wait(1)\n", "chest": 7, "attempt": 2}]
  carried = {"notes": "n", "draft": "def d():\n    wait(1)\n"}
  task = at.Task(chest=0, body="humanoid", mount="counter", grader="door")
  job = ep.job_for(task)
  mind = NotesMind('def a():\n    wait(1)\n')
  pr = mp.Practice(mind, "a-mind", st.hand_facts(), job, thing=ep.THINGS["door"])
  rows = ep.practice_on(0, pr, 1, kept=kept, carried=carried, write_notes=True, picture=False,
                        build=stub_door_world, job="door")
  # graded as a door, in its own time
  assert rows[0]["task"]["grader"] == "door" and rows[0]["run"]["budgetS"] == task.limit_s
  assert rows[0]["verdict"]["reason"].startswith("the door never opened")
  # every word the mind was sent names the cupboard, never the lid's box
  said = _asked(mind)
  assert "a cupboard like this one" in said and "passed cupboard 7 on its try 2" in said
  assert "your last cupboard" in said and "start this cupboard from" in said
  assert "THIS CUPBOARD IS DONE: none of your tries passed" in said
  assert not re.search(r"\b(box|lid|chest|knob)\b", said.lower())
  # the lid's, as they were
  lid = NotesMind('def a():\n    wait(1)\n')
  pr = mp.Practice(lid, "a-mind", st.hand_facts(), ep.JOB.format(limit=30))
  ep.practice_on(0, pr, 1, kept=kept, carried=carried, write_notes=True, picture=False,
                 budget_s=30.0, build=lambda t: stub_world(t))
  assert lid.asked[0]["messages"][0]["content"].startswith(LID_SHOWN)
  assert lid.asked[-1]["messages"][-1]["content"] == LID_NOTES_ASK.format(
    how="none of your tries passed", cap=mp.NOTES_CHARS)


def test_the_doors_job_is_its_graders_and_names_no_way_to_do_it():
  """Written before any mind flew it, as the lid's was: what stands there
  and the criteria as `challenge/door.py` judges them -- 50 deg at the call
  and the hold's end, nothing of the robot's on the door through the hold,
  in its own time -- and no word of how."""
  from pluggybot.challenge import door
  from pluggybot.evaluation import attempts as at
  task = at.Task(chest=0, body="humanoid", mount="counter", grader="door")
  job = ep.job_for(task)
  assert "cupboard" in job and "lever" in job and "26 mm" in job and "tag 55" in job
  assert f"at least {math.degrees(door.OPEN_RAD):.0f} degrees" in job and "both\ntimes" in job
  assert f"again\n{door.HOLD_S:.0f} s later" in job and f"{task.limit_s:.0f} s at most" in job
  for word in ("press", "latch", "pull", "push", "hinge", "grip", "swing", "turn"):
    assert not re.search(rf"\b{word}", job.lower()), word
  # its time limit is its own; the lid's, the drawer's and the cube's stay
  assert task.limit_s == at.JOB_LIMIT_S["door"] == 3 * at.DOOR_REFERENCE_S
  for grader in ("lid", "drawer", "place"):
    assert at.Task(chest=0, body="humanoid", mount="counter", grader=grader).limit_s == \
      3 * at.REFERENCE_S


#: The jobs as the robots are told them since #544, their templates' sha256:
#: every flight's prompt starts from one, so it changes only on purpose,
#: pinned again here.
JOB_SHA = {"lid": "ab4facadbd286f2e50d36f1fd47c6842b80953d40e5992dc7cc51a7886988b7d",
           "door": "e9e6e4821a092d14137683c9502cfcb0f8ce1b7b407d511f077074793dd49c67",
           "table": "92cd112ee170316c6667f29db651dab898a3e5cd384456dcb75da675fb7e6064"}
#: ...and what each job's fall sentence says ends however it does.
ENDS = {"lid": "lid", "door": "door", "table": "place"}


def test_every_job_tells_the_robot_a_fall_fails_the_try():
  """Ben, 2026-10-25 (#544): a fall fails the try, used or by accident, and
  the robot is told -- in every job a practice flies, over the try and the
  hold its grader reads (`lid.judged`), and never the old rule's lost time.
  The texts are pinned again, deliberately."""
  import hashlib
  from pluggybot.challenge import lid
  from pluggybot.evaluation import attempts as at
  assert set(JOB_SHA) == set(ep.JOBS)
  for grader, template in ep.JOBS.items():
    job = " ".join(ep.job_for(at.Task(chest=0, body="humanoid", mount="counter",
                                      grader=grader)).split())
    assert (f"If you fall during a try or in the {lid.HOLD_S:.0f} s after it, the try fails, "
            f"however the {ENDS[grader]} ends.") in job, grader
    assert "loses you the time" not in job, grader
    assert hashlib.sha256(template.encode()).hexdigest() == JOB_SHA[grader], grader


def test_a_verb_an_if_tests_is_read_and_the_procedure_goes_on():
  """Its failing is the test's answer, recorded and shown -- and a failed
  verb that no `if` tests still ends the procedure there."""
  body = _stub()
  body.out_of_reach = {"left"}
  facts = st.hand_facts()
  src = ('def a():\n    if not reach("left", 0.3, 0.2, 0.9):\n'
         '        reach("right", 0.3, -0.2, 0.9)\n    x = read("right.x")\n')
  ran = body.run(lang.run_procedure_routine(_Life(body), lang.compile_procedure(src, facts),
                                            facts))
  assert ran["ok"] and "failedAt" not in ran and ran["locals"]["x"] == pytest.approx(0.3)
  left, right = ran["steps"]
  assert not left["ok"] and left["tested"] and right["ok"] and "tested" not in right
  text = mp.report_text(1, ran, {"ok": False, "reason": "no"}, 1.0, False, 1)
  assert "line 2 reach: FAILED, read by its `if`, and the procedure went on" in text
  assert "and ended." in text
  # ...a test that holds runs its body; and an untested failure still stops
  ran = body.run(lang.run_procedure_routine(_Life(body), lang.compile_procedure(
    'def a():\n    if reach("right", 0.3, -0.2, 0.9):\n        wait(1)\n'
    '    reach("left", 0.3, 0.2, 0.9)\n    wait(2)\n', facts), facts))
  assert [s["verb"] for s in ran["steps"]] == ["reach", "wait", "reach"]
  assert ran["failedAt"] == 2 and not ran["ok"]
  # ...and the mind is told of it wherever it may write one: everywhere the
  # hands' vocabulary is (#525)
  assert "A verb may also stand alone as the test" in mp.system_prompt(facts)


def test_a_run_whose_last_call_is_a_test_that_read_false_reached_its_end_and_says_why():
  """#629: `completed` was the number of the last call that did what it
  said, so a run whose last call was a test that read false said "(1
  step)" in History, nothing of why, and its verdict "1/2 steps, stopped
  (error)" -- the served robots' runs of 82 calls read "2 steps" so, and
  they built theories on it. Its reason is said, and it reached its end."""
  from pluggybot import lifecycle as lc
  from pluggybot.economy import scoring
  body = _stub()
  body.out_of_reach = {"right"}
  facts = st.hand_facts()
  src = 'def a():\n    wait(1)\n    if reach("right", 0.3, -0.2, 0.9):\n        wait(1)\n'
  ran = body.run(lang.run_procedure_routine(_Life(body), lang.compile_procedure(src, facts),
                                            facts))
  assert ran["ok"] and ran["completed"] == ran["total"] == 2 and ran["testsFalse"] == 1
  why = ran["steps"][-1]["reason"]
  assert "right" in why
  assert lc.procedure_outcome("a", ran) == [
    f"ran the procedure a to its end (2 steps; 1 test read false -- line 3, reach: {why})"]
  ok, _, said = scoring.eval_program(scoring.sample_program(None, None, {"procedure": ran}, {}))
  assert ok and said == "a: ran to its end, 2 steps; 1 test read false"


def test_a_carried_hand_may_turn_as_it_goes_written_as_a_reach_turns_it():
  """The reference's first lesson -- past 50 deg the hand turns as the lid
  does -- had no verb: `move_hand` kept the turn it began with. Given
  fingers and jaws as `reach` takes them, it turns the hand to them as it
  goes; left 0, the turn is kept; half a turn is refused. A pull takes no
  turn (#525): its turn is what what it holds gives it."""
  body = _stub()
  body.start_at(2.0, 1.0, math.pi / 2)
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n'
             '    move_hand("left", 0.3, 0.2, 1.0)\n'
             '    move_hand("left", 0.3, 0.2, 1.1, fx=1, jy=1)\n'
             '    move_hand("left", 0.3, 0.2, 1.15, fz=-1, jy=1)\n', body)
  assert ran["ok"]
  (s1, v1, t1), (s2, v2, t2) = body.turned
  assert (s1, v1, s2, v2) == ("left", "move_hand", "left", "move_hand")
  # the fingers ahead of the stance -- the world's +y, facing up the y axis
  # -- and then straight down: each the turn's third column, reversed
  assert np.allclose(-np.asarray(t1)[:, 2], [0.0, 1.0, 0.0], atol=1e-9)
  assert np.allclose(-np.asarray(t2)[:, 2], [0.0, 0.0, -1.0], atol=1e-9)
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n'
             '    move_hand("left", 0.3, 0.2, 1.0, fx=1)\n', _stub())
  assert ran["steps"][-1]["reason"].startswith("give both ways")
  with pytest.raises(st.Refused):
    lang.compile_procedure('def a():\n    pull("left", 0, 0, 1, 10, 0.05, fz=-1, jy=1)\n',
                           st.hand_facts())


def test_the_humanoids_pull_hands_its_way_to_the_reach_in_the_stance_frame():
  """`Body.pull_routine` takes its way in the believed world frame; the
  reach's force move (`Reach.pull`) works in the stance frame, with the
  force, the distance, the speed and whether it gives as they were asked."""
  from pluggybot.evaluation import attempts as at
  world = at.set_out(at.Task(chest=0, body="humanoid", mount="counter"))
  try:
    body = world.body
    got = {}

    def pull(side, way, force, distance_m, speed=0.06, give=True):
      got.update(side=side, way=np.asarray(way, dtype=float), force=force,
                 distance=distance_m, speed=speed, give=give)
      return False
    body.reach.out = lambda side: True
    body.reach.pull = pull
    yaw = body.stance[2]
    body.run(body.pull_routine("left", (math.cos(yaw), math.sin(yaw), 0.0), 12.0, 0.1,
                               speed=0.05, give=False))
    assert got["side"] == "left" and np.allclose(got["way"], [1.0, 0.0, 0.0], atol=1e-9)
    assert (got["force"], got["distance"], got["speed"], got["give"]) == (12.0, 0.1, 0.05, False)
    body.run(body.pull_routine("left", (0.0, 0.0, 2.0), 5.0, 0.2))
    assert np.allclose(got["way"], [0.0, 0.0, 1.0]) and got["give"] is True
  finally:
    world.close()


def test_a_path_is_its_points_carried_as_one_motion_each_point_turned_as_it_asks():
  """`via` adds a point and moves nothing; the next `move_hand` carries the
  hand through them all on its way to its own end in one move (the body's
  path, one posture: separate moves set the reference's own arc down on 2
  chests of 8), a point with no turn keeping the one before, and empties
  the path (#525 merged #493's `follow` into it)."""
  body = _stub()
  facts = st.hand_facts()
  src = ('def a():\n    reach("left", 0.3, 0.2, 0.9)\n'
         '    via("left", 0.3, 0.2, 0.95)\n'
         '    via("left", 0.35, 0.2, 1.0, fx=1, jy=1)\n'
         '    move_hand("left", 0.4, 0.2, 1.05)\n    x = read("left.x")\n')
  life = _Life(body)
  ran = body.run(lang.run_procedure_routine(life, lang.compile_procedure(src, facts), facts))
  assert ran["ok"] and ran["locals"]["x"] == pytest.approx(0.4)
  assert not life.paths.get("left") and ran["steps"][-1]["points"] == 2
  side, path, pos, turns = body.carried_paths[-1]
  assert side == "left" and len(path) == 2 and pos == pytest.approx((0.4, 0.2, 1.05))
  # the first point keeps the hand's own turn, the second takes the one
  # given, and the end the one before -- None, as `Reach.carry` reads it
  assert len(turns) == 3 and turns[0] is None and turns[1] is not None and turns[2] is None
  # ...a path with no turns keeps the hand's, and a move with no path is a line
  body.run(lang.run_procedure_routine(_Life(body), lang.compile_procedure(
    'def a():\n    reach("left", 0.3, 0.2, 0.9)\n    via("left", 0.3, 0.2, 1.0)\n'
    '    move_hand("left", 0.3, 0.2, 1.1)\n', facts), facts))
  assert body.carried_paths[-1][3] is None and len(body.carried_paths[-1][1]) == 1
  body.run(lang.run_procedure_routine(_Life(body), lang.compile_procedure(
    'def a():\n    reach("left", 0.3, 0.2, 0.9)\n    move_hand("left", 0.3, 0.2, 1.0)\n',
    facts), facts))
  assert body.carried_paths[-1][1] == []
  # ...and past the cap, it says so
  life = _Life(body)
  life.paths = {"left": [((0.0, 0.0, 0.0), None)] * st.MAX_VIA}
  ran = body.run(lang.run_procedure_routine(life, lang.compile_procedure(
    'def a():\n    via("left", 0.3, 0.2, 0.9)\n', facts), facts))
  assert f"a path holds {st.MAX_VIA} points at most" in ran["steps"][-1]["reason"]
  # ...and the mind is told of the maths and the path wherever there are hands
  prompt = mp.system_prompt(facts)
  assert "`sin(a)`, `cos(a)` (a in radians)" in prompt and "via(" in prompt
  assert "trigonometry." not in prompt and "follow(" not in prompt
