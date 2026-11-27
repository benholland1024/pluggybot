"""`face` (issue #765): the verb turned as `step` turns its feet, closed on
the stance -- each rule pinned as cheaply as it fails (docs/Testing.md). The
flights that measured it are SimNotes' "`face` turned as `step` turns".

  1. its turn is the step's (`Body.step_routine`, its turn alone), closed on
     the STANCE's own heading: a turn of 10 deg or more ends within the
     step's tolerance where the feet lag the pelvis -- the fake walk of
     `tests/test_step.py` -- where the loop's own turn, closed on the pelvis,
     read ok with them short;
  2. flown (a few seconds of sim, the humanoid on a bare floor): the stance
     ends within a degree or two of a turn of 45 deg each way, its legs back
     on the standing policy, and 4 s later has turned back no further than
     `SETTLE_BACK_DEG`;
  3. one that does not get there reads false saying how, off its reasons,
     and its text says it turns as `step` does.
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
  (`HumanoidMission.face_routine`: the physics step, its tolerance), so a
  verb turned by it is judged here on the stance it leaves."""

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
  body.run(body.hold_routine(1.0))
  a = body._true_stance()
  heading = math.remainder(body.stance[2] + math.radians(deg), 2 * math.pi)
  rep = body.run(st._face(ProcedureLife(body), {"heading": heading}))
  landed = math.degrees(math.remainder(body._true_stance()[2] - a[2], 2 * math.pi)) - deg
  body.run(body.mission.to_stand_routine())
  body.run(body.hold_routine(4.0))
  held = math.degrees(math.remainder(body._true_stance()[2] - a[2], 2 * math.pi)) - deg
  body.close()
  assert rep["ok"] and body.mission.falls == 0, rep
  said = f"landed {landed:+.2f} deg off the turn, {held:+.2f} 4 s after its legs stood"
  assert abs(landed) <= math.degrees(hb.TURN_MISS_RAD), said
  assert abs(held) <= SETTLE_BACK_DEG, said


@pytest.mark.parametrize("why, said", [
  ("turned short", "your turn stopped short of heading 1.20 rad"),
  ("out of time", "did not square up to heading 1.20 rad within its time"),
  ("fell", "you fell")])
def test_a_face_that_does_not_get_there_reads_false_saying_how(why, said):
  """Its turn's end in its own words (#765): `step`'s say "your turn", a
  face names the heading it was asked."""
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
