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

#: The leg driver's position field, rad a count: 16 bits over +-12.5 rad.
LEG_POSITION_LSB = 25.0 / 65535
#: ...and its velocity field, rad/s a count: 12 bits over +-65 rad/s.
LEG_VELOCITY_LSB = 130.0 / 4095


def quantised(x, lsb: float):
  """A value sent in a fixed-point field: the nearest count."""
  return np.round(np.asarray(x) / lsb) * lsb
