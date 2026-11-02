"""A humanoid standing still keeps its heading (issue #593, its second fix).

Settled at rest -- resting, and `HOLD_AFTER_S` past its last motion command
-- its odometry turns its heading by nothing while the IMU's own heading
agrees that nothing turns the body, and learns the gyro's offset
(`imu.HeadingHold`, #425's zero-rate update on the heading alone). Unheld,
wherever no scan was matched, Rowan's offset (3 deg a minute) turned its
heading past the matcher's 6 deg reach in two minutes; standing in `DONE`,
105-161 deg.

Each rule pinned without a flight (docs/Testing.md): the odometry fed a
heading and a clock, the hold's own arithmetic in `test_sensor_noise.py`,
and the one line of wiring on a body that steps nothing. The flights that
measured the settle are SimNotes' "A heading held at a standstill".
"""

import json
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from pluggybot.body import body_for
from pluggybot.humanoid import mission as hu
from pluggybot.humanoid import odometry as od
from pluggybot.mapping import scan_match
from pluggybot.perception import imu

from test_humanoid_body import _floor  # noqa: I001 -- tests/ is on sys.path

#: The odometry's tick here, s: the served one's is 10 ms, and nothing in
#: the hold depends on it but the smoothing, which 50 ms keeps.
TICK_S = 0.05
#: A body's sway in heading as it stands, rad and Hz: at its smoothed rate,
#: 0.045 deg/s, about what a settled humanoid read (p99 0.05).
SWAY, SWAY_HZ = 0.0002, 1.0


def _standing(settled_after_s: float, offset: float = imu.GYRO_BIAS):
  """The odometry of a body standing still on both soles, swaying, its
  IMU's heading turning by `offset` rad/s on top of the sway, on a clock
  the test holds; settled once the clock passes `settled_after_s`. Returns
  it, the clock, and the true heading at the clock's time."""
  clock = [0.0]

  def truth() -> float:
    return SWAY * math.sin(2 * math.pi * SWAY_HZ * clock[0])

  odo = od.StanceOdometry(NS(soles=[0, 1]), NS(yaw=lambda: truth() + offset * clock[0]),
                          clock=lambda: clock[0],
                          settled=lambda: clock[0] >= settled_after_s)
  odo.z = 0.9
  mids = np.array([[0.0, 0.1, -0.9], [0.0, -0.1, -0.9]])

  def soles(q, g):
    odo.corners = np.array([[m + [a, b, 0.0] for a in (-0.1, 0.1) for b in (-0.04, 0.04)]
                            for m in mids])
    odo.flat = np.array([True, True])
    odo.low = odo.corners[:, :, 2].min(axis=1)
    return mids

  odo._soles = soles
  return odo, clock, truth


def _stand(odo, clock, truth, seconds: float) -> float:
  """Tick it `seconds` on; the worst its heading was off the truth, deg."""
  worst = 0.0
  for _ in range(round(seconds / TICK_S)):
    clock[0] += TICK_S
    odo.step(None, None)
    off = math.remainder(odo.yaw - truth(), 2 * math.pi)
    worst = max(worst, abs(math.degrees(off)))
  return worst


def test_standing_settled_on_the_parts_worst_offset_the_heading_stays_in_the_matchers_reach():
  """The issue's acceptance: standing still on a biased gyro, five minutes
  where no scan is matched, the heading off by what the offset turned it
  before it settled (1.5 deg at the part's worst) and no more -- the
  matcher reaches 6 -- and the offset learned. Unheld, 15 deg."""
  odo, clock, truth = _standing(hu.HOLD_AFTER_S)
  worst = _stand(odo, clock, truth, 300.0)
  before = math.degrees(imu.GYRO_BIAS) * hu.HOLD_AFTER_S
  assert worst == pytest.approx(before, abs=0.05)
  assert worst < math.degrees(scan_match.SEARCH_RAD)
  assert odo.hold.bias == pytest.approx(imu.GYRO_BIAS, rel=0.02)
  odo, clock, truth = _standing(math.inf)
  assert _stand(odo, clock, truth, 300.0) == pytest.approx(15.0, abs=0.05), \
    "the premise: unheld, its offset turned it"


def test_the_body_settles_past_its_settle_resting_and_tells_its_odometry():
  """The wiring: settled is RESTING (no walk of its own, no step aside, no
  hand out, on its feet) and `HOLD_AFTER_S` past its last motion command
  -- the settle a stopped humanoid turns through, measured 0.34 deg/s 20 s
  after a turn on the spot -- and the odometry asks it, and the sim's
  clock, every tick, its drivers fresh or not."""
  body = body_for(*_floor())
  try:
    m = body.mission
    m.data.time = 100.0
    m.last_motion_t = 100.0 - hu.HOLD_AFTER_S + 0.1
    assert m.resting and not m._settled(), "resting, not yet past the settle"
    m.last_motion_t = 100.0 - hu.HOLD_AFTER_S
    assert m._settled()
    for what, back in (("legs", "stand"), ("driving", 0)):
      setattr(m, what, {"legs": "getup", "driving": 1}[what])
      assert not m._settled(), what
      setattr(m, what, back)
    m.reach.phase["left"] = "reach"
    assert not m._settled(), "a hand out"
    m.reach.phase["left"] = "rest"
    body.start_at(0.0, 0.0, 0.0)
    m.data.time = 100.0
    assert m.odo.settled == m._settled and m.odo.clock() == 100.0
  finally:
    body.close()


def test_the_hold_rides_a_restart_and_a_save_from_before_it_starts_one_anew():
  """#345's rule: what decides is kept -- the offset learned, its smoothed
  rate and the heading's last tick -- so a restored body turns as the one
  that went on; and the deployed save, from before the hold, restores with
  nothing learned."""
  a, clock, truth = _standing(hu.HOLD_AFTER_S)
  _stand(a, clock, truth, 60.0)
  state = json.loads(json.dumps(a.kept_state()))
  b, _, _ = _standing(hu.HOLD_AFTER_S)
  b.clock, b.att, b.settled = a.clock, a.att, a.settled
  b.restore_kept(state)
  for _ in range(100):
    clock[0] += TICK_S
    a.step(None, None)
    b.step(None, None)
  assert (b.offset, b.hold.bias) == (a.offset, a.hold.bias)
  old = {k: v for k, v in state.items() if k not in ("hold", "headingAt")}
  c, _, _ = _standing(hu.HOLD_AFTER_S)
  c.restore_kept(old)
  assert c.hold.bias == 0.0 and c.offset == state["offset"]
