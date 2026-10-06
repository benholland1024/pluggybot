"""Encoders (issue #386): what a joint's angle reads as, once it is counted.
The sim's joint positions are exact; a robot's are whole counts.

  the quadruped's legs the GDS68 in MIT mode reports each joint over CAN in
                       fixed-width fields: position in 16 bits and velocity
                       in 12 (the MIT protocol Steadywin's manual points
                       to). The fields' ranges are set in the driver and
                       the part's sheets do not publish them; stated here
                       as the protocol's defaults in Katz's firmware --
                       +-12.5 rad and +-65 rad/s, the second above the
                       joint's 31.4 rad/s no-load speed, which a
                       narrower range would clip. The 14-bit encoder
                       itself resolves 48 urad at the output, finer than
                       the field it is sent in. Backlash is on top
                       (`legs/actuator.py`).
"""

from __future__ import annotations

import numpy as np

from pluggybot.legs.actuator import GIM8108_8

#: The leg driver's position field, rad a count: 16 bits over +-12.5 rad.
LEG_POSITION_LSB = 25.0 / 65535
#: ...and its velocity field, rad/s a count: 12 bits over +-65 rad/s.
LEG_VELOCITY_LSB = 130.0 / 4095


def quantised(x, lsb: float):
  """A value sent in a fixed-point field: the nearest count."""
  return np.round(np.asarray(x) / lsb) * lsb


#: The arm's drivers report each motor's torque too (issue #407): the MIT
#: reply's third field, 12 bits over the driver's torque range -- set to the
#: motor's peak (`legs.actuator.GIM8108_8`: +-22 N*m) -- read off the
#: measured phase current (Iq * Kt). The current it is read off is noisy,
#: and no Steadywin sheet publishes how: stated as a FOC driver's current
#: sense at rest, 25 mA RMS, times the motor's output Kt (1.19 N*m/A).
CURRENT_NOISE_A = 0.025
TORQUE_RANGE_NM = GIM8108_8.peak_torque
TORQUE_LSB = 2 * TORQUE_RANGE_NM / 4095
TORQUE_NOISE_NM = CURRENT_NOISE_A * GIM8108_8.kt


def torque_readings(tau, key: str) -> np.ndarray:
  """`torque_reading` along a run of steps at once, its noise ONE stream
  keyed on `key` (an imagination's rollout, `imagination.rollout`): the same
  spread and counts, without a generator a step (40 % of a rollout)."""
  import zlib
  tau = np.asarray(tau, dtype=float)
  rng = np.random.default_rng(zlib.crc32(f"torques:{key}".encode()))
  noisy = tau + rng.normal(0.0, TORQUE_NOISE_NM, tau.shape)
  top = int(TORQUE_RANGE_NM / TORQUE_LSB)
  return np.clip(np.round(noisy / TORQUE_LSB), -top, top) * TORQUE_LSB


def torque_reading(tau: float, key: str, step: int) -> float:
  """What a driver reports of a motor's torque `tau` at physics step `step`:
  its noise (deterministic on the step and `key`, a robot's motor -- a
  crc32 seed, never `hash()`) and its field's counts, saturating at the
  last count inside the range."""
  import zlib
  seed = zlib.crc32(f"torque:{key}:{step}".encode())
  noisy = float(tau) + float(np.random.default_rng(seed).normal(0.0, TORQUE_NOISE_NM))
  top = int(TORQUE_RANGE_NM / TORQUE_LSB)
  return float(np.clip(np.round(noisy / TORQUE_LSB), -top, top) * TORQUE_LSB)
