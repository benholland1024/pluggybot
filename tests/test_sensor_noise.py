"""Honest sensors (issue #386): what the robot is told by its IMU and its
encoders, off the parts' datasheets (docs/Parts.md), never the sim's exact
joint angles and rates. Each rule pinned with a direct call."""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.perception import encoders, imu


def test_the_gyro_reads_its_datasheet_noise_density():
  # 0.0028 deg/s/sqrt(Hz): per sample of dt, sigma = density / sqrt(dt),
  # so an integrated angle walks by density * sqrt(t) -- whatever dt is.
  part = imu.Imu("test:noise")
  dt = 0.002
  samples = np.array([part.gyro_z(0.0, dt) for _ in range(20000)])
  samples -= part.gyro_bias[2]
  assert samples.std() == pytest.approx(imu.GYRO_NOISE / math.sqrt(dt), rel=0.05)


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


def test_two_robots_never_draw_one_stream_and_one_robot_draws_the_same_twice():
  a, b, again = imu.Imu("rover"), imu.Imu("r2_rover"), imu.Imu("rover")
  assert a.gyro_z(0.1, 0.002) != b.gyro_z(0.1, 0.002)
  assert imu.Imu("rover").gyro_z(0.1, 0.002) == again.gyro_z(0.1, 0.002)


def test_a_restart_carries_the_imu_on_where_it_stopped():
  a = imu.Imu("rover")
  for _ in range(100):
    a.gyro_z(0.2, 0.002)
  b = imu.Imu("something else")
  b.restore_kept(a.kept_state())
  assert [a.gyro_z(0.2, 0.002) for _ in range(5)] == [b.gyro_z(0.2, 0.002) for _ in range(5)]


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


def test_a_push_while_resting_is_turned_through_not_held():
  # The zero-rate update (#425) holds the heading only while the gyro agrees
  # that nothing turns the body: another robot shoving a lying one 30 deg
  # round is integrated, within the gyro's scale error and the offset it
  # has not learned yet -- and the rest is taken up again after it.
  part, still = imu.Imu("test:push"), imu.Standstill()
  att, dt = imu.Attitude((1.0, 0.0, 0.0, 0.0)), 0.002

  def lie(seconds, rate_z=0.0):
    for _ in range(round(seconds / dt)):
      att.step(still.rate(part.gyro((0.0, 0.0, rate_z), dt), True, dt),
               (0.0, 0.0, imu.G), dt)

  lie(10.0)
  assert still.still and att.yaw() == 0.0
  lie(3.0, math.radians(10.0))
  assert not still.still
  assert math.degrees(att.yaw()) == pytest.approx(30.0, abs=0.3)
  lie(2.0)
  assert still.still


def test_a_slide_too_slow_to_see_teaches_no_more_than_the_part_could_be_off():
  # Lying, a turn under the gate is missed and learned as the gyro's offset
  # (#425) -- none was measured on the floor or the dock, but nothing turns
  # it back until the next rest: unbounded, a 0.12 deg/s slide for 2 min
  # taught 0.17 deg/s, and every walk after it turned 7 deg a minute.
  part, still, dt = imu.Imu("test:slide"), imu.Standstill(), 0.002
  part.gyro_bias[:] = (0.0, 0.0, imu.GYRO_BIAS)
  for _ in range(round(120.0 / dt)):
    still.rate(part.gyro((0.0, 0.0, math.radians(0.12)), dt), True, dt)
  assert still.still, "the premise: a slide the gate cannot see"
  assert max(abs(b) for b in still.bias) <= imu.BIAS_MAX


def test_a_wheel_encoder_reads_whole_counts():
  # 64 counts a motor turn on the Pololu #4753, 3200 at the 50:1 output.
  step = 2 * math.pi / encoders.WHEEL_COUNTS_PER_REV
  assert encoders.counted(3.5 * step) == pytest.approx(3 * step)
  assert encoders.counted(-0.5 * step) == pytest.approx(-step)
  assert encoders.counted(10.0) <= 10.0 < encoders.counted(10.0) + step


def test_the_leg_driver_reports_in_its_fields_counts():
  # 16 bits over +-12.5 rad, 12 over +-65 rad/s (the MIT protocol's fields).
  assert encoders.LEG_POSITION_LSB == pytest.approx(0.3815e-3, rel=1e-3)
  q = np.array([0.1234567, -1.2])
  got = encoders.quantised(q, encoders.LEG_POSITION_LSB)
  assert np.all(np.abs(got - q) <= encoders.LEG_POSITION_LSB / 2)
  assert np.allclose(got / encoders.LEG_POSITION_LSB, np.round(got / encoders.LEG_POSITION_LSB))


def test_the_rovers_reckoner_is_told_counted_wheels_and_its_imus_rate():
  # The wiring, not the physics: after a step the reckoner's last wheel
  # angles are whole counts, and its heading moved by what the IMU read.
  from pluggybot.rack.swap import HubSwap
  model = mujoco.MjModel.from_xml_path("models/room_hub.xml")
  data = mujoco.MjData(model)
  swap = HubSwap(model, data)
  read = []
  real = swap.imu.gyro_z
  swap.imu.gyro_z = lambda rate, dt: read.append(real(rate, dt)) or read[-1]
  swap.reckoner.update(*swap.encoders())
  data.qpos[swap.left_adr] += 0.1234
  swap.step((0.0, 0.3))
  step = 2 * math.pi / encoders.WHEEL_COUNTS_PER_REV
  assert swap.reckoner._prev_left / step == pytest.approx(round(swap.reckoner._prev_left / step))
  assert read, "the heading was integrated off the sim's gyro, not the IMU"
  assert swap.reckoner.theta == pytest.approx(read[-1] * model.opt.timestep)


def test_the_quadruped_attaches_with_its_joint_ranges_in_radians():
  # The include form carries no <compiler>: parsed on its own, MuJoCo read
  # the knee's -2.75..-0.35 rad as degrees, and the robot attached into the
  # home world was thrown into the air by its own joint limits.
  from pluggybot.legs.model import CHOSEN, attachable, body_xml
  alone = mujoco.MjModel.from_xml_string(body_xml(CHOSEN))
  spec = mujoco.MjSpec()
  spec.attach(attachable(CHOSEN), prefix="", frame=spec.worldbody.add_frame())
  attached = spec.compile()
  for name in ("FL_hip_abd", "FL_hip_flex", "FL_knee"):
    assert np.allclose(attached.jnt_range[attached.joint(name).id],
                       alone.jnt_range[alone.joint(name).id])
