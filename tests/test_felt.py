"""The whole body's reading of a hand's force (issue #547, stage 2): each rule
pinned on the robot's model of itself with torques staged on it -- what its
drivers would read with a force at its grip -- or on a bare reach; the
flights that measured it are `scripts/humanoid_spike.py --feel` and `--bar`.

  1. the legs feel along a straight arm, which the arm alone cannot;
  2. a joint on its stop is left out (its stop carries what its motor does
     not), and a balance that does not close reads nothing;
  3. a twist at the grip is read as a twist, not a push;
  4. a hand is read off the whole body while it is at work and the
     standing policy has the legs, else off its arm alone -- and the
     mission says which has the legs.
"""

from types import SimpleNamespace as NS

import mujoco
import numpy as np
import pytest

from pluggybot.humanoid import arms as ha
from pluggybot.humanoid import friction
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import reach as rc

from test_hands import _bare_reach

#: The left arm held straight out ahead at a counter's height, its elbow
#: 0.05 rad off its stop.
STRAIGHT = {"shoulder_pitch": -1.45, "shoulder_roll": 0.1, "shoulder_yaw": 0.0, "elbow": 0.05,
            "wrist_pitch": 0.0, "wrist_roll": 0.0}


def _posed(arm=STRAIGHT):
  sm = ha.SelfModel()
  q = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
  for j, v in arm.items():
    q[hm.JOINT_NAMES.index(f"left_{j}")] = v
  sm.pose(q, np.array([0.0, 0.0, -1.0]))
  return sm, q


def _staged(sm, push, twist=(0.0, 0.0, 0.0)):
  """Every joint's torque as its driver would read it, still, with the left
  hand pushing `push` (N) and twisting `twist` (N*m) on what it holds and
  the soles holding the rest: G = S^T tau + sum J^T w, the soles' share the
  least that balances the trunk."""
  m, d = sm.m, sm.d
  d.qvel[:] = 0.0
  mujoco.mj_comVel(m, d)
  bias = np.zeros(m.nv)
  mujoco.mj_rne(m, d, 0, bias)
  jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
  mujoco.mj_jacSite(m, d, jp, jr, sm.grip["left"])
  outside = jp.T @ -np.asarray(push, float) + jr.T @ -np.asarray(twist, float)
  soles = []
  for g in sm.soles:
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jp, jr, d.geom_xpos[g], m.geom_bodyid[g])
    soles += [jp, jr]
  j = np.vstack(soles)
  w = np.linalg.lstsq(j[:, :6].T, (bias - outside)[:6], rcond=None)[0]
  return (bias - outside - j.T @ w)[sm.vadr]


def _still(sm):
  return friction.still(friction.load(), sm.m.dof_frictionloss[sm.vadr])


@pytest.mark.parametrize("way", [(1, 0, 0), (-1, 0, 0), (1, 0, -1), (0, 0, 1)])
def test_the_legs_feel_along_a_straight_arm_where_the_arm_cannot(way):
  """A load along a nearly straight arm goes through its structure, not its
  motors: pushed 20 N away, the arm's own reading saw 5 and 2 away and
  down (#547). The whole body's balance reads it off the legs."""
  sm, q = _posed()
  push = 20.0 * np.asarray(way, float) / np.linalg.norm(way)
  tau = _staged(sm, push)
  sm.pose(q, np.array([0.0, 0.0, -1.0]))
  felt = sm.felt(["left"], tau, _still(sm))["left"]
  assert np.allclose(felt, push, atol=0.1)
  if way[0] > 0:
    # the premise: pushing away along it, the arm alone reads under half
    along = float(sm.hand_force("left", tau[sm.arm["left"]]) @ push) / 20.0
    assert along < 10.0


def test_a_joint_on_its_stop_is_left_out_of_the_reading():
  """Pushed along, an arm straightened onto its elbow's stop, which held 3.1
  N*m its motor never saw: read in, the hand read 4 N long and 9 N up."""
  sm, q = _posed({**STRAIGHT, "elbow": 0.0})
  push = np.array([20.0, 0.0, 0.0])
  tau = _staged(sm, push)
  tau[hm.JOINT_NAMES.index("left_elbow")] += 3.1
  sm.pose(q, np.array([0.0, 0.0, -1.0]))
  assert np.allclose(sm.felt(["left"], tau, _still(sm))["left"], push, atol=0.1)


def test_a_twist_at_the_grip_is_read_as_a_twist_not_a_push():
  """A grip twists what it holds (on the hands check's grips 0.7-1.7 N*m on
  average): taken as a force alone, a twist of 1 N*m read 3 N of push."""
  sm, q = _posed()
  tau = _staged(sm, np.zeros(3), twist=(0.0, 1.0, 0.0))
  sm.pose(q, np.array([0.0, 0.0, -1.0]))
  assert np.linalg.norm(sm.felt(["left"], tau, _still(sm))["left"]) < 0.5


def test_a_balance_that_does_not_close_reads_nothing():
  """Dragged onto a sole's edge by `--hold`'s hardest pull, the legs read
  the hand's pull 60 N the wrong way while its arm read the truth: where
  the legs and the arm disagree past `CLOSES`, the whole body reads
  nothing (and the reach keeps the arm's, below)."""
  sm, q = _posed()
  arm = sm.arm["left"]
  tau = _staged(sm, np.array([0.0, -40.0, 0.0]))
  tau[arm] = _staged(sm, np.array([0.0, 20.0, 0.0]))[arm]
  assert sm.felt(["left"], tau, _still(sm)) == {}


def _feeling(phase, standing, felt=2.0):
  r = _bare_reach()
  n = len(hm.JOINT_NAMES)
  r.tau, r.qd, r.friction, r.still = np.zeros(n), np.zeros(n), {}, np.zeros(n)
  r.force = {s: np.zeros(3) for s in hm.SIDES}
  r.self_model.hand_force = lambda side, tau: np.array([1.0, 0.0, 0.0])
  r.self_model.felt = lambda hands, tau, still: (
    {} if felt is None else {s: np.array([felt, 0.0, 0.0]) for s in hands})
  r.phase["left"], r.standing = phase, standing
  r._feel()
  return r


@pytest.mark.parametrize("phase", rc.FELT_PHASES)
def test_a_hand_at_work_standing_is_read_off_the_whole_body(phase):
  r = _feeling(phase, standing=True)
  assert r.force["left"][0] == 2.0 and r.arm_force["left"][0] == 1.0
  # ...and a hand at rest beside it off its arm
  assert r.force["right"][0] == 1.0


@pytest.mark.parametrize("phase", ["rest", "move", "joints", "home"])
def test_a_hand_not_at_work_is_read_off_its_arm(phase):
  assert _feeling(phase, standing=True).force["left"][0] == 1.0


def test_a_balance_that_reads_nothing_leaves_the_arms_reading():
  assert _feeling("hold", standing=True, felt=None).force["left"][0] == 1.0


def test_a_walking_body_reads_its_hands_off_their_arms():
  """The balance has both soles planted: walking, a swinging leg's own
  torques read up to 400 N at a sole (`humanoid.odometry`)."""
  assert _feeling("hold", standing=False).force["left"][0] == 1.0


@pytest.mark.parametrize("legs, standing", [("stand", True), ("walk", False)])
def test_the_mission_says_whether_the_standing_policy_has_the_legs(legs, standing):
  from pluggybot.humanoid.mission import HumanoidMission
  m = object.__new__(HumanoidMission)
  m.getup, m.legs = None, legs
  m.walker = NS(ask=lambda *a: None, step=lambda: None)
  m.stand = NS(step=lambda: None)
  m.reach = NS(step=lambda: None, standing=None)
  m._before_step((0.0, 0.0, 0.0))
  assert m.reach.standing is standing
