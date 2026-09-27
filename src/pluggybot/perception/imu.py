"""The IMU (issue #386): a TDK ICM-42688-P, on both bodies (docs/Parts.md),
read with the noise, the offset left after calibration and the scale error
its datasheet gives -- DS-000347 rev 1.9, Tables 1 and 2. The sim's own
gyro and accelerometer sensors are perfect; this is what the robot is told.

Every number is the datasheet's, or a stated range with its reason:

  noise      white, at the rate noise density (tested in production)
  offset     the zero-rate / zero-g offset LEFT AFTER the boot calibration:
             the datasheet's variation over temperature across
             `TEMP_SWING_C`, a stated swing (the initial +-0.5 deg/s and
             +-20 mg are what a calibration at rest removes); constant
             over a life, drawn uniformly in +-it per axis
  scale      the sensitivity's initial tolerance, +-0.5 % (tested in
             production), drawn uniformly per axis

Not modelled: cross-axis sensitivity (+-1.25 % gyro, +-1 % accel) and
nonlinearity (+-0.1 %), which on a body that stays near level add a
fraction of a percent of rates that average to nothing.

`Attitude` turns the two into roll and pitch (a complementary filter) and a
heading, the way the robot would; the quadruped's odometry reads it.
"""

from __future__ import annotations

import math
import zlib

import numpy as np

G = 9.80665
#: Rate noise spectral density, rad/s/sqrt(Hz): 0.0028 deg/s/sqrt(Hz).
GYRO_NOISE = math.radians(0.0028)
#: The temperature change between the boot calibration and now, degC: a
#: stated choice -- a board warming up beside a Pi 5 and its motors.
TEMP_SWING_C = 10.0
#: ZRO variation vs temperature, +-0.005 deg/s/degC, over the swing: the
#: gyro's offset after calibration, rad/s (0.05 deg/s).
GYRO_BIAS = math.radians(0.005 * TEMP_SWING_C)
#: Gyro sensitivity scale factor initial tolerance, +-0.5 %.
GYRO_SCALE = 0.005
#: Accelerometer noise density, m/s^2/sqrt(Hz): 65 ug/sqrt(Hz) on x and y,
#: 70 on z.
ACCEL_NOISE = (65e-6 * G, 65e-6 * G, 70e-6 * G)
#: Zero-g level change vs temperature, +-0.15 mg/degC, over the swing: the
#: accelerometer's offset after calibration, m/s^2 (1.5 mg).
ACCEL_BIAS = 0.15e-3 * G * TEMP_SWING_C
#: Accelerometer sensitivity scale factor initial tolerance, +-0.5 %.
ACCEL_SCALE = 0.005


def seed_for(key: str) -> int:
  """A sensor's seed, off a checksum of its name and robot -- never
  `hash()`, which Python salts per process."""
  return zlib.crc32(key.encode())


class Imu:
  """One ICM-42688-P. `key` names it (the robot's prefix and the part), so
  two robots never draw one stream. Rates and forces are in the body
  frame; `dt` is the step they were sampled over."""

  def __init__(self, key: str) -> None:
    self.rng = np.random.default_rng(seed_for(f"imu:{key}"))
    u = self.rng.uniform
    self.gyro_bias = u(-GYRO_BIAS, GYRO_BIAS, 3)
    self.gyro_scale = 1.0 + u(-GYRO_SCALE, GYRO_SCALE, 3)
    self.accel_bias = u(-ACCEL_BIAS, ACCEL_BIAS, 3)
    self.accel_scale = 1.0 + u(-ACCEL_SCALE, ACCEL_SCALE, 3)
    self._accel_noise = np.array(ACCEL_NOISE)

  def gyro(self, rate, dt: float) -> np.ndarray:
    """What the gyro reads for a true body rate, rad/s."""
    n = self.rng.standard_normal(3) * (GYRO_NOISE / math.sqrt(dt))
    return np.asarray(rate) * self.gyro_scale + self.gyro_bias + n

  def gyro_z(self, rate_z: float, dt: float) -> float:
    """The yaw axis alone, for a body that reads nothing else (the
    rover's heading): one draw a step, not three."""
    n = float(self.rng.standard_normal()) * (GYRO_NOISE / math.sqrt(dt))
    return (float(rate_z) * float(self.gyro_scale[2])
            + float(self.gyro_bias[2]) + n)

  def accel(self, force, dt: float) -> np.ndarray:
    """What the accelerometer reads for a true specific force, m/s^2."""
    n = self.rng.standard_normal(3) * (self._accel_noise / math.sqrt(dt))
    return np.asarray(force) * self.accel_scale + self.accel_bias + n

  def kept_state(self) -> dict:
    """The part's draws and where its noise had got to (issue #345: a
    restart saves the noise generators' STATE)."""
    return {"rng": self.rng.bit_generator.state,
            "gyroBias": self.gyro_bias.tolist(),
            "gyroScale": self.gyro_scale.tolist(),
            "accelBias": self.accel_bias.tolist(),
            "accelScale": self.accel_scale.tolist()}

  def restore_kept(self, state: dict) -> None:
    self.rng.bit_generator.state = state["rng"]
    self.gyro_bias = np.array(state["gyroBias"])
    self.gyro_scale = np.array(state["gyroScale"])
    self.accel_bias = np.array(state["accelBias"])
    self.accel_scale = np.array(state["accelScale"])


#: The complementary filter's time constant, s: how long the accelerometer
#: takes to pull the tilt back to gravity. Long enough that a trot's bob
#: (~3 Hz) and a start's push average out, short enough that the gyro's
#: offset holds the tilt off by only its product with this (0.05 deg).
TAU_S = 1.0


class Attitude:
  """Roll and pitch held by the gyro and pulled toward the accelerometer's
  gravity (a complementary filter, `TAU_S`), heading from the gyro alone:
  the body's orientation as its IMU would say it. Starts from `quat`
  (w, x, y, z; world from body), the orientation the body is known to
  boot in. Plain floats: it runs every physics step, and a numpy 3-vector
  costs a microsecond an operation."""

  def __init__(self, quat) -> None:
    self.q = tuple(float(v) for v in quat)

  def step(self, gyro, accel, dt: float) -> None:
    """Advance by one sample: the measured body rate and specific force."""
    w, x, y, z = self.q
    gx, gy, gz = (float(v) for v in gyro)
    fx, fy, fz = (float(v) for v in accel)
    norm = math.sqrt(fx * fx + fy * fy + fz * fz)
    if norm > 1e-6:
      # At rest the accelerometer reads +g along the world's up, in the
      # body frame; the estimate's own up is R^T e_z. Turning the body
      # about (measured x estimated) at 1/TAU_S walks the estimate's up to
      # the measured one -- about a level axis, so the heading is left be.
      ux, uy, uz = 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
      mx, my, mz = fx / norm, fy / norm, fz / norm
      k = 1.0 / TAU_S
      gx += k * (my * uz - mz * uy)
      gy += k * (mz * ux - mx * uz)
      gz += k * (mx * uy - my * ux)
    ax, ay, az = gx * dt, gy * dt, gz * dt
    angle = math.sqrt(ax * ax + ay * ay + az * az)
    if angle > 0.0:
      s = math.sin(angle / 2) / angle
      c = math.cos(angle / 2)
      bx, by, bz = ax * s, ay * s, az * s
      # q (x) dq: a turn in the BODY frame.
      w, x, y, z = (w * c - x * bx - y * by - z * bz,
                    w * bx + x * c + y * bz - z * by,
                    w * by - x * bz + y * c + z * bx,
                    w * bz + x * by - y * bx + z * c)
      n = math.sqrt(w * w + x * x + y * y + z * z)
      self.q = (w / n, x / n, y / n, z / n)

  def turn(self, dyaw: float) -> None:
    """Turn the estimate about the WORLD's up by `dyaw`: a heading corrected
    from outside, the tilt untouched."""
    c, s = math.cos(dyaw / 2), math.sin(dyaw / 2)
    w, x, y, z = self.q
    self.q = (c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w)

  def matrix(self) -> np.ndarray:
    """World from body."""
    w, x, y, z = self.q
    return np.array([
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

  def yaw(self) -> float:
    """The heading: the body's forward axis on the floor."""
    w, x, y, z = self.q
    return math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))

  def level(self) -> np.ndarray:
    """The tilt with the heading taken out: level from body."""
    c, s = math.cos(self.yaw()), math.sin(self.yaw())
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ self.matrix()

  def tilt(self) -> float:
    """The angle between the body's up and the world's, rad."""
    w, x, y, z = self.q
    return math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y))))
