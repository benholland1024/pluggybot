"""Encoders (issue #386): what a joint's angle reads as, once it is counted.
The sim's joint positions are exact; a robot's are whole counts of the
fixed-width fields its drivers send them in (each driver's own,
`humanoid.parts.Fields`).
"""

from __future__ import annotations

import numpy as np


def quantised(x, lsb: float):
  """A value sent in a fixed-point field: the nearest count."""
  return np.round(np.asarray(x) / lsb) * lsb
