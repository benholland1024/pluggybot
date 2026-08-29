"""Guards for the hub tool-coupling spike (rack/coupling.py, milestone-8 prep).

These pin the MEASURED envelope. If a geometry change makes the "outside the
envelope" test start passing, that is good news wearing a test failure --
re-measure and update SimNotes, don't just delete the assert.
"""

from pluggybot.rack.coupling import run_cycle, run_pick


def test_aligned_cycle_picks_and_returns():
  res, _ = run_cycle()
  assert res["picked"], "aligned pick failed -- the latch mechanics broke"
  assert res["returned"], "aligned return failed -- the hang-up mechanics broke"
  # Contact stays at the approach push cap; a jam shows up as a spike (the
  # sweep measured 30-120 N on out-of-envelope trials).
  assert res["max_force_n"] < 15.0


def test_lateral_envelope_holds_at_pm_4mm():
  for dy in (-0.004, 0.004):
    res, _ = run_cycle(dy=dy)
    assert res["picked"] and res["returned"], \
      f"cycle failed inside the measured +/-4 mm lateral envelope (dy={dy})"


def test_yaw_4deg_is_outside_the_envelope():
  """Yaw is the tight axis (again): returns fail beyond ~2 deg. This guards
  the KNOWN LIMITATION -- navigation must deliver <2 deg at the hub, which
  FACE_OUTLET's measured 0.5 deg settle already does."""
  res, _ = run_cycle(yaw_deg=4.0)
  assert not res["returned"], \
    "yaw=4deg cycle succeeded: the envelope improved -- re-measure and " \
    "update SimNotes/PluggyPlan"


def test_retention_exceeds_base_traction():
  """The carried tool must survive more shake than the traction-limited base
  can produce (~4-7 m/s^2); measured to hold through 8."""
  res, _ = run_pick(shake_accel=6.0)
  assert res["picked"] and res["carried"], "tool dropped inside the base's accel range"
