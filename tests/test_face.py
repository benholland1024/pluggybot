"""`face` (issue #765): the verb turned as `step` turns its feet, closed on
the stance -- each rule pinned as cheaply as it fails (docs/Testing.md). The
flights that measured it are SimNotes' "`face` turned as `step` turns".

  1. its turn is the step's (`Body.step_routine`, its turn alone), closed on
     the STANCE's own heading: a turn of 10 deg or more ends within the
     step's tolerance where the feet lag the pelvis -- the fake walk of
     `tests/test_step.py`; so, as a step's, a correction to the believed
     heading while it turns is not followed, and a turn over `TURN_STOP_RAD`
     hands the legs to the walker first, one under it moving nothing (#765's
     review);
  2. flown (a few seconds of sim, the humanoid on a bare floor): the stance
     ends within a degree or two of a turn of 45 deg each way, its legs back
     on the standing policy, and 4 s later has turned back no further than
     `SETTLE_BACK_DEG`;
  3. one that does not get there reads false saying how, off its reasons;
     one whose legs went back before both soles read down says so, as a
     step's (#740); and its text says where its feet end, as `step`'s says
     its turns, never naming `step`.
(Its turn asked from where the stance faces in the map, #754's line: in
`tests/test_humanoid_words.py`.)
"""

import math
from types import SimpleNamespace as NS

import pytest

from pluggybot.humanoid import body as hb
from pluggybot.humanoid.mission import HumanoidMission
from pluggybot.procedure import steps as st

from test_guard import _floor_body
from test_step import DT, TURN_STEP, Walk, _walker


class Turned(Walk):
  """The fake walk carrying what the loop's own turn reads too
  (`HumanoidMission.face_routine`: the physics step, its tolerance): the
  verb before #765 called it, so this file run against that verb judges it
  on the stance it leaves (and fails there)."""

  model = NS(opt=NS(timestep=DT))
  FACING_TOLERANCE = HumanoidMission.FACING_TOLERANCE
  face_routine = HumanoidMission.face_routine


def _faced(walk, heading: float) -> dict:
  """The verb `face(heading)` on the humanoid's body over a fake walk,
  driven to its end, its clock the walk's."""
  body = _walker(walk)
  body.arm_kept, body.carrying, body.arm_held = (lambda: None), (lambda s: False), (lambda s: False)
  routine = st._face(NS(body=body), {"heading": heading})
  try:
    while True:
      next(routine)
      body._d.time = walk.t
  except StopIteration as e:
    return e.value


@pytest.mark.parametrize("feet, deg", [(0.25, 10.0), (0.25, -20.0), (0.4, 20.0), (0.4, -10.0)])
def test_a_face_of_ten_degrees_or_more_ends_within_the_steps_tolerance_on_the_stance(feet, deg):
  """#765: the walker yaws its pelvis over planted feet (#649), so a turn
  closed on the pelvis's heading -- the loop's own, which the verb was --
  read ok with the feet 6-10 deg short of a turn of 10 deg or more (505 of
  505 in the pass's census, 392 of 392 of the robots' own in the house).
  Turned as `step` turns, the stance ends within `TURN_TOL_RAD` of the
  heading. Here the fake's feet follow a share of the pelvis's turn."""
  walk = Turned(feet=feet)
  heading = math.radians(deg)
  rep = _faced(walk, heading)
  assert rep["ok"], rep
  assert walk.stance_yaw == pytest.approx(heading, abs=hb.TURN_TOL_RAD + 2 * TURN_STEP), (
    f"the stance ended {math.degrees(walk.stance_yaw - heading):+.2f} deg off the heading")
  assert walk.legs == "stand", "its legs back on the standing policy"


#: How far a face's stance may turn back in the 4 s after its legs stand,
#: deg (#765, flown: from the boot, 0.0-3.1 at +-45 deg; turned by the
#: loop's own turn, 9.3-12.3 short by then, and turning back still).
SETTLE_BACK_DEG = 5.0


@pytest.mark.parametrize("deg", [45.0, -45.0])
def test_a_face_flown_lands_its_feet_and_they_stay_once_its_legs_stand(deg):
  """#765, flown: the humanoid on a bare floor, stood 1 s from its boot,
  asked `face` 45 deg from where its stance believes it faces; its legs then
  handed to the standing policy as the next verb would (a no-op where the
  verb left them there), and held 4 s, where a `spot` that bows or a `reach`
  acts. The true stance (the soles' mean heading) lands within a degree or
  two of the turn, and turns back no further than `SETTLE_BACK_DEG` --
  turned by the loop's own turn, it landed 3-4 deg short and was 9-12 short
  4 s later."""
  from pluggybot.evaluation.practice import ProcedureLife
  body = _floor_body()
  try:
    body.run(body.hold_routine(1.0))
    a = body._true_stance()
    heading = math.remainder(body.stance[2] + math.radians(deg), 2 * math.pi)
    rep = body.run(st._face(ProcedureLife(body), {"heading": heading}))
    landed = math.degrees(math.remainder(body._true_stance()[2] - a[2], 2 * math.pi)) - deg
    body.run(body.mission.to_stand_routine())
    body.run(body.hold_routine(4.0))
    held = math.degrees(math.remainder(body._true_stance()[2] - a[2], 2 * math.pi)) - deg
  finally:
    body.close()
  assert rep["ok"] and body.mission.falls == 0, rep
  said = f"landed {landed:+.2f} deg off the turn, {held:+.2f} 4 s after its legs stood"
  assert abs(landed) <= math.degrees(hb.TURN_MISS_RAD), said
  assert abs(held) <= SETTLE_BACK_DEG, said


def test_a_face_turns_by_its_legs_and_follows_no_correction_made_meanwhile():
  """#765's review: a face's turn is the step's, closed on its legs' own
  reckoning (`HumanoidBody._by_legs`), so a correction to the believed
  heading while it turns -- a scan match, a relocation -- is not followed:
  its legs turn the turn asked from where the stance believed it faced, the
  belief ends off the heading by the correction, and the verb reads ok. It
  never mattered flown: over #765's 1,385 faces the belief's heading moved
  at most 0.6 deg against the truth while one ran (SimNotes, "`face`
  turned as `step` turns")."""
  walk = Turned()
  walk.correct_at, walk.correction = 100, (0.0, 0.0, 0.1)
  heading = math.radians(20.0)
  rep = _faced(walk, heading)
  assert rep["ok"], rep
  legs = walk.pose[2] - walk.odo.corrected[2]
  assert legs == pytest.approx(heading, abs=hb.TURN_TOL_RAD + 2 * TURN_STEP), (
    f"its legs turned {math.degrees(legs - heading):+.2f} deg off the turn asked")
  assert walk.stance[2] - heading == pytest.approx(0.1, abs=hb.TURN_TOL_RAD + 2 * TURN_STEP), (
    "the turn followed the correction")


def _handovers(walk) -> list:
  """How many commands the fake walk had been asked each time its legs were
  handed to the walker (`Walk.to_walk_routine`)."""
  seen, real = [], walk.to_walk_routine

  def to_walk():
    if walk.legs != "walk":
      seen.append(len(walk.asked))
    yield from real()
  walk.to_walk_routine = to_walk
  return seen


@pytest.mark.parametrize("deg", [0.3, 2.0, -3.5])
def test_a_face_of_a_few_degrees_hands_its_legs_to_the_walker_and_turns(deg):
  """#765's review: a face's turn is the step's, so a turn over
  `TURN_STOP_RAD` (0.5 deg) hands the legs to the walker first -- the walk's
  hand-over, which brings a hand out at work home or carries what it holds,
  as every walk verb's does -- where the verb before it did nothing under its
  4 deg tolerance; one under it moves nothing. It never arose flown: of the
  92 faces asked under 4 deg in #765's flights, every one began with its legs
  on the walker or both hands at rest (SimNotes, "`face` turned as `step`
  turns"). Here the fake's feet follow a quarter of the pelvis's turn, as
  `tests/test_step.py`'s small turns' do."""
  walk = Turned(feet=0.25)
  handed = _handovers(walk)
  heading = math.radians(deg)
  rep = _faced(walk, heading)
  assert rep["ok"], rep
  turns = [k for k, (_, _, w) in enumerate(walk.asked) if w]
  if abs(heading) <= hb.TURN_STOP_RAD:
    assert not turns and not handed and walk.stance_yaw == 0.0, (turns[:3], handed)
    return
  assert turns and handed and handed[0] <= turns[0], (turns[:3], handed)
  assert walk.stance_yaw == pytest.approx(heading, abs=hb.TURN_TOL_RAD + 2 * TURN_STEP), (
    f"the stance ended {math.degrees(walk.stance_yaw - heading):+.2f} deg off the heading")
  assert walk.legs == "stand"


@pytest.mark.parametrize("why, said", [
  ("turned short", "your turn stopped short of heading 1.20 rad"),
  ("out of time", "did not square up to heading 1.20 rad within its time"),
  ("fell", "you fell"),
  ("blocked", "did not turn to heading 1.20 rad")])
def test_a_face_that_does_not_get_there_reads_false_saying_how(why, said):
  """Its turn's end in its own words (#765): `step`'s say "your turn", a
  face names the heading it was asked -- and an end a turn alone has no
  words for (none today: only the three above) names the heading and
  guesses no cause (#765's review: it read "stopped short")."""
  def step_routine(ahead, left=0.0, turn=0.0):
    return {"why": why}
    yield
  body = NS(stance=(0.0, 0.0, 0.0), arm_kept=lambda: None, carrying=lambda s: False,
            arm_held=lambda s: False, step_routine=step_routine)
  routine = st._face(NS(body=body), {"heading": 1.2})
  with pytest.raises(StopIteration) as done:
    next(routine)
  assert done.value.value == {"ok": False, "why": why, "reason": said}


def test_a_face_whose_legs_went_back_before_both_soles_read_down_says_so():
  """#740's hand-back is a face's too: where its settle's bound ran out first
  the verdict says so (`feetDown`: false), as a step's does, and nothing
  where both soles read down."""
  def faced(settled):
    def step_routine(ahead, left=0.0, turn=0.0):
      return {"why": "stepped", **({} if settled else {"feetDown": False})}
      yield
    body = NS(stance=(0.0, 0.0, 0.0), arm_kept=lambda: None, carrying=lambda s: False,
              arm_held=lambda s: False, step_routine=step_routine)
    with pytest.raises(StopIteration) as done:
      next(st._face(NS(body=body), {"heading": 0.5}))
    return done.value.value
  assert faced(False) == {"ok": True, "feetDown": False}
  assert faced(True) == {"ok": True}


def test_faces_text_says_where_its_feet_end_as_steps_says_its_turns():
  """Its line says what it now does (#765): a turn in place to a heading in
  the map (#754), ok once the feet are turned to it within the step's own
  bound (`TURN_MISS_RAD`, "a degree or two", as `step`'s text says its
  turns), its legs standing again; one that does not get there reads false.
  ⚠ It never names `step`: a world whose verbs have none is told none
  (`tests/test_hands.py`)."""
  doc = st.VERBS["face"].doc
  assert "rad in your map" in doc and "`step`" not in doc
  assert "your feet are turned to it within a degree or two" in doc
  assert math.degrees(hb.TURN_MISS_RAD) <= 2.0
  assert "your legs stand again" in doc and "reads false and says so" in doc
  assert "A turn lands within a degree or two of what you asked" in st.HANDS_VERBS["step"].doc
