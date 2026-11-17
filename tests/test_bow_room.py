"""#717, Ben's option 2: a look's bow is judged against what stands in front
of the head. Unjudged, `spot`'s 0.45 rad bow at the lobby's wall pushed the
head into it at 11-17 N, the wall threw the trunk back, the fall watch handed
the legs to the get-up, and it ran the robot 2.45 m backward into the far
wall: a stuck death."""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from pluggybot.body import StubBody
from pluggybot.humanoid import body as hb
from pluggybot.humanoid import model as hm
from pluggybot.humanoid.arms import SelfModel
from pluggybot.humanoid.reach import statue
from pluggybot.procedure import steps as st

from test_hands import _bare, _fly


def _standing(wall_x: float | None, t: float = 10.0, peer_x: float | None = None):
  """A body standing upright in its stance frame, its newest scan at its
  head's height a wall across its way at stance x `wall_x` (none: nothing),
  taken just now from its sensor over its pelvis -- and, `peer_x`, another
  robot's body that scan saw straight ahead there."""
  sm = SelfModel()
  statue(sm, (0.0, 0.0, 0.0), np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES]))
  px = float(sm.d.qpos[0])
  a = np.radians(np.arange(-80.0, 81.0))
  r = np.full(len(a), 8.0) if wall_x is None else (wall_x - px) / np.cos(a)
  pa = np.radians(np.arange(-10.0, 11.0))
  peers = (t, pa, (peer_x - px) / np.cos(pa)) if peer_x is not None else (t, pa[:0], pa[:0])
  mission = NS(over_returns=(t, a, r, (0.0, 0.0)), peer_returns=peers, odo=NS(sm=sm),
               last_motion_t=0.0, scans=[], look_peers=None)
  body = NS(mission=mission, _d=NS(time=t), reach=NS(posture=np.zeros(3)))
  mission.look_scan = lambda: mission.scans.append(body._d.time)
  return body


def test_a_bow_is_held_where_the_head_would_come_too_near_a_wall():
  """The deepest bow whose head's front (its model of itself's statue) stays
  `BOW_CLEAR_M` short of the wall: all of it where the wall is far, held
  shallower where it is near, none where the head is that near already."""
  front = {b: hb.head_front(0.0, b) for b in (0.0, 0.25, 0.45)}
  assert front[0.0] < front[0.25] < front[0.45], front
  far = _standing(front[0.45] + hb.BOW_CLEAR_M + 0.01)
  assert hb.HumanoidBody.bow_room(far, 0.45)[0] == pytest.approx(0.45)
  near = _standing(front[0.25] + hb.BOW_CLEAR_M + 0.005)
  bow, at = hb.HumanoidBody.bow_room(near, 0.45)
  assert bow == pytest.approx(0.25) and hb.head_front(0.0, bow + hb.BOW_STEP_RAD) + hb.BOW_CLEAR_M > at
  assert hb.HumanoidBody.bow_room(_standing(front[0.0] + hb.BOW_CLEAR_M - 0.01), 0.45)[0] == 0.0
  assert hb.HumanoidBody.bow_room(_standing(None), 0.45)[0] == pytest.approx(0.45)
  # ...and another robot's body in front of its head is as near a thing
  near_peer = _standing(None, peer_x=front[0.25] + hb.BOW_CLEAR_M + 0.005)
  assert hb.HumanoidBody.bow_room(near_peer, 0.45)[0] == pytest.approx(0.25)


def test_a_bow_is_judged_off_a_scan_taken_where_it_stands():
  """Standing still its LIDAR takes no scan of its own: a scan from before
  it last walked, or older than `BOW_SCAN_S`, is no reading of what stands in
  front of its head now, and one is taken where it stands (#717's review)."""
  body = _standing(None, t=10.0)
  hb.HumanoidBody.bow_room(body, 0.45)
  assert body.mission.scans == [], "a fresh scan, taken again"
  body.mission.last_motion_t = 10.2
  body._d.time = 10.4
  hb.HumanoidBody.bow_room(body, 0.45)
  assert body.mission.scans == [10.4], "a scan from before it walked, read"
  body = _standing(None, t=10.0)
  body._d.time = 10.0 + hb.BOW_SCAN_S + 0.1
  hb.HumanoidBody.bow_room(body, 0.45)
  assert body.mission.scans == [body._d.time]


def test_the_heads_front_off_the_model_of_itself_is_the_lobbys_measure():
  """M12 flew the lobby's stuck start: the head cap met the wall at a 0.25
  rad bow from 0.12-0.14 m off the LIDAR, and the 0.45 bow pushed it in. Its
  model of itself puts the head's front past both."""
  assert hb.head_front(0.0, 0.25) - hb.head_front(0.0, 0.0) > 0.14 - 0.072
  assert hb.head_front(0.0, 0.0) == pytest.approx(0.058, abs=0.01)


def test_a_look_that_did_not_see_its_tag_says_how_its_bow_was_held():
  """The robot reads why it saw nothing: what stood in front of its head,
  and how far it bowed of what it would have, or that it could not bow."""
  for held, said in (({"asked": 0.45, "bow": 0.25, "m": 0.12},
                      "your LIDAR reads something 0.12 m in front of your head, so you bowed "
                      "only 14 degrees of the 26 you would have"),
                     ({"asked": 0.25, "bow": 0.0, "m": 0.03},
                      "your LIDAR reads something 0.03 m in front of your head, too near for "
                      "you to bow at all")):
    body = StubBody(*_bare())
    body.bow_held = held
    ran = _fly('def a():\n    spot(55)\n', body)
    assert ran["steps"][0]["reason"].endswith("; " + said), ran["steps"][0]["reason"]
  body = StubBody(*_bare())
  ran = _fly('def a():\n    spot(55)\n', body)
  assert "your LIDAR reads" not in ran["steps"][0]["reason"]


def test_the_spots_text_says_how_far_a_bow_keeps_the_head_off_its_constant():
  doc = st.HANDS_VERBS["spot"].doc
  assert f"leaves your head {100 * hb.BOW_CLEAR_M:.0f} cm short of what your LIDAR reads at your " \
         "head's height in front of you" in doc
  assert math.isclose(hb.BOW_CLEAR_M, 0.05)
