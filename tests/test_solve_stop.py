"""`SelfModel.solve` ends where its joints come back bit for bit to joints it
held (#769): an iteration is a function of the joints alone, so from there it
cycles and its cap's joints are known -- returned, and the model posed, as
the cap would leave them (SimNotes, "A solve ended where its joints come
back"). Each rule pinned on the robot's model of itself, against the solve
before #769 copied line for line (`_to_the_cap`), as cheaply as it fails.
"""

import math

import numpy as np
import pytest

from pluggybot.humanoid import arms
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import reach

TURNS = {"fwd": arms.FORWARD, "down": arms.DOWN, "none": None}

#: Targets on the model of itself posed at the stand afresh, each from the
#: arm hanging ("hang") or a seed of `reach.SEEDS`, and how the solve before
#: #769 ran there: its joints' first return, (at iteration, to iteration),
#: else how it ended -- fixed points, cycles of 2 at either phase and of 6 at
#: one other than its first, a position-only solve, one that converges and one
#: capped with no return (C18's grid, `c18-review/p2/explore_repose.py`). ⚠ The rest of
#: the model is part of the iteration: the other arm posed elsewhere, its
#: last bits part (a Jacobian is read through the body's centre of mass).
CASES = [
  ("left", (0.15, -0.45, 1.65), "fwd", "hang", 40, (22, 21)),
  ("left", (0.15, -0.15, 1.65), "fwd", 2, 80, (51, 50)),
  ("left", (0.15, -0.45, 0.25), "down", 4, 80, (5, 3)),
  ("left", (0.15, -0.45, 1.3), "down", 5, 80, (24, 22)),
  ("right", (0.15, -0.15, 0.95), "fwd", 4, 80, (13, 11)),
  ("left", (0.15, -0.45, 0.25), "fwd", 0, 80, (59, 53)),
  ("left", (0.55, -0.15, 0.25), "down", "hang", 40, (15, 13)),
  ("left", (0.15, -0.45, 1.65), "none", 5, 80, (59, 57)),
  ("left", (0.15, -0.15, 0.95), "none", 0, 80, "converges"),
  ("left", (0.15, -0.45, 0.25), "fwd", "hang", 40, "capped"),
]


def _to_the_cap(sm, side, pos, rot, iters):
  """The solve before #769, line for line, to its tolerance or its cap; and
  the iteration its joints first came back to joints it held, with the
  iteration they came back to (None if never), read beside it."""
  idx = sm.arm[side]
  lo, hi = sm.ranges[idx, 0] + arms.RANGE_MARGIN, sm.ranges[idx, 1] - arms.RANGE_MARGIN
  q = np.clip(sm.q[idx], lo, hi)
  sm._set_arm(side, q)
  ep, er = np.inf, np.inf
  seen, back = {q.tobytes(): 0}, None
  for k in range(1, iters + 1):
    have_p, have_r = sm.grip_pose(side)
    e_p = np.asarray(pos, dtype=float) - have_p
    ep = float(np.linalg.norm(e_p))
    if rot is None:
      e, j, er = e_p, sm.jacobian(side)[:3], 0.0
    else:
      e_r = arms._rot_error(np.asarray(rot), have_r)
      er = float(np.linalg.norm(e_r))
      e = np.concatenate([e_p, arms.IK_ROT_WEIGHT * e_r])
      j = sm.jacobian(side)
      j[3:] *= arms.IK_ROT_WEIGHT
    if ep < arms.IK_POS_TOL and er < arms.IK_ROT_TOL:
      break
    dq = j.T @ np.linalg.solve(j @ j.T + arms.IK_DAMPING ** 2 * np.eye(len(e)), e)
    dq = np.clip(dq, -arms.IK_STEP, arms.IK_STEP)
    q = np.clip(q + dq, lo, hi)
    if back is None:
      at = seen.setdefault(q.tobytes(), k)
      if at < k:
        back = (k, at)
    sm._set_arm(side, q)
  have_p, have_r = sm.grip_pose(side)
  ep = float(np.linalg.norm(np.asarray(pos) - have_p))
  er = 0.0 if rot is None else float(np.linalg.norm(arms._rot_error(np.asarray(rot), have_r)))
  return q, ep, er, back


STAND = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
UPRIGHT = np.array([0.0, 0.0, -1.0])


@pytest.fixture(scope="module")
def stood():
  """The model of itself, every solve's iterations counted (its Jacobian is
  read once an iteration)."""
  sm = arms.SelfModel()
  ran, read = [0], sm.jacobian

  def counted(side):
    ran[0] += 1
    return read(side)
  sm.jacobian = counted
  return sm, None, ran


def _from(sm, home, side, start):
  """Posed at the stand afresh, the arm hanging or at a seed."""
  sm.pose(STAND, UPRIGHT)
  if start != "hang":
    sm.q[sm.arm[side]] = reach.seed(side, reach.SEEDS[start])


def _left(sm, side):
  """What the solve leaves the model posed as: its joints, its qpos and its
  grip's place, bit for bit."""
  return sm.q.tobytes(), sm.d.qpos.tobytes(), sm.d.site_xpos[sm.grip[side]].tobytes()


def _both(sm, home, ran, side, pos, rot, start, iters, again=False):
  """The solve and the one before #769 from the same arm: each one's answer,
  how it left the model, and its iterations -- `again`, each solved twice,
  the second from where the first left the arm."""
  out = []
  for solver in (lambda *a: _to_the_cap(sm, *a), sm.solve):
    _from(sm, home, side, start)
    if again:
      solver(side, pos, rot, iters)
    ran[0] = 0
    got = solver(side, pos, rot, iters)
    out.append((got, _left(sm, side), ran[0]))
  return out


def _same(cap, new):
  (q0, ep0, er0, *_), left0, _ = cap
  (q1, ep1, er1), left1, _ = new
  assert q1.tobytes() == q0.tobytes() and ep1 == ep0 and er1 == er0, "its answer is not the cap's"
  assert left1 == left0, "it left the model posed otherwise than the cap does"


@pytest.mark.parametrize("side, pos, turn, start, iters, ran", CASES)
def test_a_solve_ends_where_its_joints_come_back_with_what_its_cap_would_return(stood, side, pos, turn,
                                                                                start, iters, ran):
  """#769: the solve before it ran to its cap long after its joints had come
  back to where they had been -- a fixed point, or a cycle at its step's clip
  -- and every iteration there changed nothing it returns. Now it ends at the
  first return, with the joints the cap would end on (a cycle's member at the
  cap's phase), the model posed as the cap leaves it, its errors read there;
  one that converges or never comes back runs as it did."""
  sm, home, count = stood
  cap, new = _both(sm, home, count, side, np.array(pos), TURNS[turn], start, iters)
  _same(cap, new)
  back = cap[0][3]
  if ran == "converges":
    assert back is None and cap[0][1] < arms.IK_POS_TOL and new[2] == cap[2] < iters
  elif ran == "capped":
    assert back is None and cap[2] == new[2] == iters
  else:
    assert back == ran, f"the case no longer comes back where it did ({back}): pick it again off a grid"
    assert cap[2] == iters and new[2] == back[0], f"it ran {new[2]} iterations, its joints back at {back[0]}"


def test_the_stop_at_the_cap_the_start_and_no_iteration_returns_what_the_cap_would(stood):
  """Its edges: joints come back on the cap's own iteration (nothing left to
  save); a solve begun on a fixed point (the same target solved again from
  where the last left the arm: back at its first iteration, to its start); and
  no iteration at all."""
  sm, home, count = stood
  side, pos, rot = "left", np.array((0.15, -0.45, 1.65)), arms.FORWARD
  cap, new = _both(sm, home, count, side, pos, rot, "hang", 22)
  _same(cap, new)
  assert cap[0][3] == (22, 21) and cap[2] == new[2] == 22
  cap, new = _both(sm, home, count, side, pos, rot, "hang", 40, again=True)
  _same(cap, new)
  assert cap[0][3] == (1, 0) and (cap[2], new[2]) == (40, 1)
  cap, new = _both(sm, home, count, side, pos, rot, "hang", 0)
  _same(cap, new)
  assert cap[2] == new[2] == 0 and math.isfinite(new[0][1])
