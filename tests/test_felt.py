"""The whole body's reading of a hand's force (issue #547, stage 2): each rule
pinned on the robot's model of itself with torques staged on it -- what its
drivers would read with a force at its grip -- or on a bare reach; the
flights that measured it are `scripts/humanoid_spike.py --feel` and `--bar`.

  1. the legs feel along a straight arm, which the arm alone cannot;
  2. a joint on its stop is left out (its stop carries what its motor does
     not), and a balance that does not close reads nothing;
  3. a twist at the grip is read as a twist, not a push, and two hands at
     work are each read as their own;
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
#: 0.05 rad off its stop (the right's mirrors its roll, yaw and wrist roll).
STRAIGHT = {"shoulder_pitch": -1.45, "shoulder_roll": 0.1, "shoulder_yaw": 0.0, "elbow": 0.05,
            "wrist_pitch": 0.0, "wrist_roll": 0.0}
_MIRRORED = ("shoulder_roll", "shoulder_yaw", "wrist_roll")


def _posed(arm=STRAIGHT, sides=("left",)):
  sm = ha.SelfModel()
  q = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
  for s in sides:
    for j, v in arm.items():
      q[hm.JOINT_NAMES.index(f"{s}_{j}")] = -v if s == "right" and j in _MIRRORED else v
  sm.pose(q, np.array([0.0, 0.0, -1.0]))
  return sm, q


def _staged(sm, pushes, twists=None):
  """Every joint's torque as its driver would read it, still, with each hand
  of `pushes` pushing that (N) and twisting `twists`' (N*m) on what it holds
  and the soles holding the rest: G = S^T tau + sum J^T w, the soles' share
  the least that balances the trunk."""
  m, d = sm.m, sm.d
  d.qvel[:] = 0.0
  mujoco.mj_comVel(m, d)
  bias = np.zeros(m.nv)
  mujoco.mj_rne(m, d, 0, bias)
  outside = np.zeros(m.nv)
  for side, push in pushes.items():
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, sm.grip[side])
    twist = (twists or {}).get(side, np.zeros(3))
    outside += jp.T @ -np.asarray(push, float) + jr.T @ -np.asarray(twist, float)
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
  tau = _staged(sm, {"left": push})
  sm.pose(q, np.array([0.0, 0.0, -1.0]))
  felt = sm.felt(["left"], tau, _still(sm))["left"]
  assert np.allclose(felt, push, atol=0.1)
  if way[0] > 0:
    # the premise: pushing away along it, the arm alone reads under half
    along = float(sm.hand_force("left", tau[sm.arm["left"]]) @ push) / 20.0
    assert along < 10.0


def test_a_joint_on_its_stop_is_left_out_of_the_reading():
  """Pushed along, an arm straightened onto its elbow's stop, which held 3.1
  N*m its motor never saw: on the bench the hand read 11 N of 20; staged
  here, read in, it reads 4 N long and 9 N up."""
  sm, q = _posed({**STRAIGHT, "elbow": 0.0})
  push = np.array([20.0, 0.0, 0.0])
  tau = _staged(sm, {"left": push})
  tau[hm.JOINT_NAMES.index("left_elbow")] += 3.1
  sm.pose(q, np.array([0.0, 0.0, -1.0]))
  assert np.allclose(sm.felt(["left"], tau, _still(sm))["left"], push, atol=0.1)


def test_a_twist_at_the_grip_is_read_as_a_twist_not_a_push():
  """A grip twists what it holds (on the hands check's grips 0.7-1.7 N*m on
  average): taken as a force alone, a twist of 1 N*m read 3 N of push."""
  sm, q = _posed()
  tau = _staged(sm, {"left": np.zeros(3)}, {"left": (0.0, 1.0, 0.0)})
  sm.pose(q, np.array([0.0, 0.0, -1.0]))
  assert np.linalg.norm(sm.felt(["left"], tau, _still(sm))["left"]) < 0.5


def test_two_hands_at_work_are_each_read_as_their_own():
  """Each hand's force and twist are the balance's own unknowns, side by
  side: the legs feel the two together, and each arm's joints tell them
  apart."""
  sm, q = _posed(sides=hm.SIDES)
  pushes = {"left": np.array([20.0, 0.0, 0.0]), "right": np.array([0.0, -5.0, 15.0])}
  tau = _staged(sm, pushes)
  felt = sm.felt(list(hm.SIDES), tau, _still(sm))
  for s in hm.SIDES:
    assert np.allclose(felt[s], pushes[s], atol=0.1)


def test_a_balance_that_does_not_close_reads_nothing():
  """Dragged onto a sole's edge by `--hold`'s hardest pull, the legs read
  the hand's pull 60 N the wrong way while its arm read the truth: where
  the legs and the arm disagree past `CLOSES`, the whole body reads
  nothing (and the reach keeps the arm's, below)."""
  sm, q = _posed()
  arm = sm.arm["left"]
  tau = _staged(sm, {"left": np.array([0.0, -40.0, 0.0])})
  tau[arm] = _staged(sm, {"left": np.array([0.0, 20.0, 0.0])})[arm]
  assert sm.felt(["left"], tau, _still(sm)) == {}


def _feeling(phase, standing, felt=2.0, friction=None, qd=None):
  r = _bare_reach()
  n = len(hm.JOINT_NAMES)
  r.tau, r.qd, r.still = np.zeros(n), np.zeros(n) if qd is None else qd, np.zeros(n)
  r.friction = friction or {}
  r.force = {s: np.zeros(3) for s in hm.SIDES}
  r.self_model.hand_force = lambda side, tau: np.array([1.0, 0.0, 0.0])
  r.handed = []

  def felt_(hands, tau, still):
    r.handed.append(tau.copy())
    return {} if felt is None else {s: np.array([felt, 0.0, 0.0]) for s in hands}
  r.self_model.felt = felt_
  r.phase["left"], r.standing = phase, standing
  r._feel()
  return r


#: A hand AT WORK: on its way in (the reach's `BLOCKED_N`), holding, pulled,
#: stopped, carried (the force move's own loop) or on its way out -- every
#: phase named, so that dropping one from `FELT_PHASES` fails here.
AT_WORK = ("approach", "hold", "pull", "blocked", "carry", "retreat")


@pytest.mark.parametrize("phase", AT_WORK)
def test_a_hand_at_work_standing_is_read_off_the_whole_body(phase):
  r = _feeling(phase, standing=True)
  assert r.force["left"][0] == 2.0 and r.arm_force["left"][0] == 1.0
  # ...and a hand at rest beside it off its arm
  assert r.force["right"][0] == 1.0


def test_the_whole_body_is_handed_the_torques_less_a_turning_arms_friction():
  """The arm's friction, calibrated, comes off where a joint turns before
  the balance reads its row, as the arm's own reading takes it off."""
  elbow = hm.JOINT_NAMES.index("left_elbow")
  qd = np.zeros(len(hm.JOINT_NAMES))
  qd[elbow] = 0.5
  r = _feeling("hold", standing=True, qd=qd,
               friction={"left_elbow": {"coulomb": 0.1, "viscous": 0.2}})
  assert r.handed[0][elbow] == pytest.approx(-(0.1 + 0.2 * 0.5))


@pytest.mark.parametrize("phase", ["rest", "move", "joints", "home"])
def test_a_hand_not_at_work_is_read_off_its_arm(phase):
  assert _feeling(phase, standing=True).force["left"][0] == 1.0


def test_a_balance_that_reads_nothing_leaves_the_arms_reading():
  assert _feeling("hold", standing=True, felt=None).force["left"][0] == 1.0


def test_a_walking_body_reads_its_hands_off_their_arms():
  """The balance has both soles planted: walking, a swinging leg's own
  torques read up to 400 N at a sole (`humanoid.odometry`)."""
  assert _feeling("hold", standing=False).force["left"][0] == 1.0


def test_a_reach_reads_its_arms_alone_until_its_owner_says_it_stands():
  """Fail closed: a body stepping a reach while it walks (the spike's
  walkers do) must not read a sole in the air as the hand's push."""
  from pluggybot.humanoid.policy import StandDriver
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  assert rc.Reach(model, data, StandDriver(model, data)).standing is False


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
