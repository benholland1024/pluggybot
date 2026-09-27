"""Encoders (issue #386): what a joint's angle reads as, once it is counted.
The sim's joint positions are exact; a robot's are whole counts.

  the rover's wheels   the Pololu #4753's encoder: 64 counts a motor turn
                       (both edges of both channels), 3200 at the 50:1
                       output (Pololu's product page); a counter reads the
                       whole counts it has passed
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

import math

import numpy as np

#: Counts a wheel turn, the rover's 50:1 gearmotor at its output.
WHEEL_COUNTS_PER_REV = 3200
#: The leg driver's position field, rad a count: 16 bits over +-12.5 rad.
LEG_POSITION_LSB = 25.0 / 65535
#: ...and its velocity field, rad/s a count: 12 bits over +-65 rad/s.
LEG_VELOCITY_LSB = 130.0 / 4095


def counted(angle: float, per_rev: int = WHEEL_COUNTS_PER_REV) -> float:
  """An angle as a counter reads it, rad: the whole counts passed."""
  step = 2.0 * math.pi / per_rev
  return math.floor(angle / step) * step


def quantised(x, lsb: float):
  """A value sent in a fixed-point field: the nearest count."""
  return np.round(np.asarray(x) / lsb) * lsb
