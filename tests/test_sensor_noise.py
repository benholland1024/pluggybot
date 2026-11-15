"""Honest sensors (issue #386): what the robot is told by its IMU, off the
part's datasheet (docs/Parts.md), never the sim's exact rates. Each rule
pinned with a direct call. (Its joints read in their drivers' counts:
`test_humanoid.py`.)"""

import math

import numpy as np
import pytest

from pluggybot.perception import imu


def test_the_gyro_reads_its_datasheet_noise_density():
  # 0.0028 deg/s/sqrt(Hz): per sample of dt, sigma = density / sqrt(dt),
  # so an integrated angle walks by density * sqrt(t) -- whatever dt is.
  part = imu.Imu("test:noise")
  dt = 0.002
  samples = np.array([part.gyro(np.zeros(3), dt) for _ in range(20000)]) - part.gyro_bias
  assert samples.std(axis=0) == pytest.approx([imu.GYRO_NOISE / math.sqrt(dt)] * 3, rel=0.05)


def test_the_offsets_and_scales_are_drawn_inside_the_datasheets_ranges():
  # The offset left after calibration is the ZRO's (and zero-g's) variation
  # over TEMP_SWING_C; the scale, the sensitivity's initial tolerance.
  parts = [imu.Imu(f"test:{k}") for k in range(50)]
  gb = np.array([p.gyro_bias for p in parts])
  ab = np.array([p.accel_bias for p in parts])
  gs = np.array([p.gyro_scale for p in parts])
  assert imu.GYRO_BIAS == pytest.approx(math.radians(0.005 * imu.TEMP_SWING_C))
  assert imu.ACCEL_BIAS == pytest.approx(0.15e-3 * imu.G * imu.TEMP_SWING_C)
  assert np.abs(gb).max() <= imu.GYRO_BIAS and np.abs(gb).max() > 0.5 * imu.GYRO_BIAS
  assert np.abs(ab).max() <= imu.ACCEL_BIAS
  assert np.abs(gs - 1.0).max() <= imu.GYRO_SCALE


def _past_one_axis(read, twin, full, inside, past):
  """Each axis in turn driven past `full`, either way, beside a twin (the
  same part, the same draws) read inside it: that axis reads the full scale,
  the others what the twin read, bit for bit."""
  for axis in range(3):
    for sign in (1.0, -1.0):
      value = np.array(inside, float)
      value[axis] = sign * past
      got, did = read(value), twin(np.array(inside, float))
      assert got[axis] == sign * full
      rest = [k for k in range(3) if k != axis]
      assert got[rest].tobytes() == did[rest].tobytes()


def test_past_its_full_scale_an_axis_reads_it_and_the_clip_draws_nothing(monkeypatch):
  # The ICM-42688-P's widest ranges, +-16 g and +-2000 deg/s (DS-000347
  # Tables 1-2): past them its output ends, after the part's own error, so a
  # landing's 43 g reads 16 (#696). The clip draws no noise: inside full
  # scale a reading is the same bits as with no full scale at all, and a
  # flight that never passes it flies as before.
  dt = 0.002
  assert imu.ACCEL_RANGE == 16.0 * imu.G
  assert imu.GYRO_RANGE == math.radians(2000.0)
  part, twin = imu.Imu("test:range"), imu.Imu("test:range")
  _past_one_axis(lambda f: part.accel(f, dt), lambda f: twin.accel(f, dt),
                 imu.ACCEL_RANGE, np.array((3.0, -2.0, 1.0)) * imu.G, 40 * imu.G)
  _past_one_axis(lambda w: part.gyro(w, dt), lambda w: twin.gyro(w, dt),
                 imu.GYRO_RANGE, np.radians((300.0, -150.0, 40.0)), math.radians(3000.0))
  # Inside full scale by as little as the part's error leaves room for (its
  # worst scale and offset, and six sigma of its noise): the bits of the same
  # part with no full scale, and each reading draws its three normals, no more.
  def inside(full, scale, bias, noise):
    return (full - bias - 6 * noise / math.sqrt(dt)) / (1 + scale)
  g = inside(imu.GYRO_RANGE, imu.GYRO_SCALE, imu.GYRO_BIAS, imu.GYRO_NOISE)
  a = inside(imu.ACCEL_RANGE, imu.ACCEL_SCALE, imu.ACCEL_BIAS, max(imu.ACCEL_NOISE))
  rate, force = np.array((g, -g, 0.5)), np.array((a, -a, imu.G))
  part, ref, drawn = (imu.Imu("test:inside") for _ in range(3))
  with monkeypatch.context() as unbounded:
    unbounded.setattr(imu, "GYRO_RANGE", math.inf)
    unbounded.setattr(imu, "ACCEL_RANGE", math.inf)
    want = [(ref.gyro(rate, dt), ref.accel(force, dt)) for _ in range(5)]
  for gyro, accel in want:
    assert part.gyro(rate, dt).tobytes() == gyro.tobytes()
    assert part.accel(force, dt).tobytes() == accel.tobytes()
    drawn.rng.standard_normal(6)
  assert part.rng.bit_generator.state == drawn.rng.bit_generator.state


def test_two_robots_never_draw_one_stream_and_one_robot_draws_the_same_twice():
  a, b, again = imu.Imu("pluggybot"), imu.Imu("r2_pluggybot"), imu.Imu("pluggybot")
  rate = (0.0, 0.0, 0.1)
  assert not np.array_equal(a.gyro(rate, 0.002), b.gyro(rate, 0.002))
  assert np.array_equal(imu.Imu("pluggybot").gyro(rate, 0.002), again.gyro(rate, 0.002))


def test_a_restart_carries_the_imu_on_where_it_stopped():
  a, rate = imu.Imu("pluggybot"), (0.0, 0.0, 0.2)
  for _ in range(100):
    a.gyro(rate, 0.002)
  b = imu.Imu("something else")
  b.restore_kept(a.kept_state())
  assert np.array_equal([a.gyro(rate, 0.002) for _ in range(5)],
                        [b.gyro(rate, 0.002) for _ in range(5)])


def _quat(axis, angle):
  axis = np.asarray(axis, float) / np.linalg.norm(axis)
  return (math.cos(angle / 2), *(axis * math.sin(angle / 2)))


def test_the_attitude_pulls_a_tilt_to_gravity_and_leaves_the_heading_to_the_gyro():
  # Booted believing it is level, a body pitched 5 deg at rest: the
  # accelerometer's gravity walks the estimate there in a few TAU_S, and
  # the heading, which gravity cannot see, does not move.
  w, x, y, z = _quat((0, 1, 0), math.radians(5))
  up_in_body = np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
  att = imu.Attitude((1.0, 0.0, 0.0, 0.0))
  for _ in range(int(5 * imu.TAU_S / 0.002)):
    att.step((0.0, 0.0, 0.0), up_in_body * imu.G, 0.002)
  assert math.degrees(att.tilt()) == pytest.approx(5.0, abs=0.1)
  assert abs(att.yaw()) < 1e-9
  # ...and a turn, level, is the gyro's alone
  att = imu.Attitude((1.0, 0.0, 0.0, 0.0))
  for _ in range(500):
    att.step((0.0, 0.0, 1.0), (0.0, 0.0, imu.G), 0.002)
  assert att.yaw() == pytest.approx(1.0, abs=1e-9)
  assert att.tilt() < 1e-9


def test_a_heading_corrected_from_outside_leaves_the_tilt_alone():
  att = imu.Attitude(_quat((1, 0, 0), math.radians(3)))
  tilt, yaw = att.tilt(), att.yaw()
  att.turn(0.4)
  assert att.tilt() == pytest.approx(tilt, abs=1e-12)
  assert att.yaw() == pytest.approx(yaw + 0.4, abs=1e-12)


def _held(hold, part, seconds, rate_z=0.0, dt=0.01):
  """What a heading hold believes the body turned over `seconds` of the
  gyro reading `rate_z`, settled throughout, rad: each tick the IMU's own
  turn, as the humanoid's odometry hands it over (every 10 ms)."""
  turned = 0.0
  for _ in range(round(seconds / dt)):
    turned += hold.turn(part.gyro((0.0, 0.0, rate_z), dt)[2] * dt, dt, True)
  return turned


def test_a_push_while_settled_is_turned_through_not_held():
  # The zero-rate update (#425, #593) holds the heading only while the gyro
  # agrees that nothing turns the body: another robot shoving a standing one
  # 30 deg round is turned through, within the gyro's scale error and the
  # offset it has not learned yet -- and the hold is taken up again after it.
  part, hold = imu.Imu("test:push"), imu.HeadingHold()
  assert _held(hold, part, 10.0) == 0.0 and hold.still
  turned = _held(hold, part, 3.0, math.radians(10.0))
  assert not hold.still
  assert math.degrees(turned) == pytest.approx(30.0, abs=0.3)
  _held(hold, part, 2.0)
  assert hold.still


def test_a_slide_too_slow_to_see_teaches_no_more_than_the_part_could_be_off():
  # Settled, a turn under the gate is missed and learned as the gyro's
  # offset, and nothing turns it back until the next rest: unbounded, a
  # 0.12 deg/s slide for 2 min taught 0.17 deg/s (#425), and every walk
  # after it turned 7 deg a minute.
  part, hold = imu.Imu("test:slide"), imu.HeadingHold()
  part.gyro_bias[:] = (0.0, 0.0, imu.GYRO_BIAS)
  _held(hold, part, 120.0, math.radians(0.12))
  assert hold.still, "the premise: a slide the gate cannot see"
  assert abs(hold.bias) <= imu.BIAS_MAX
