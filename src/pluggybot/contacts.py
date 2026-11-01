"""The contact list, read as an ARRAY (rooftop #296): never walked struct by
struct on the physics seam.

⚠ MEASURED: the electrical criteria and the rover's two chassis scans ran
EVERY physics step, per robot, as Python loops over `data.contact[i]` -- a
pybind struct per contact, 83 contacts at rest in the home world, 500 steps
a sim-second, two robots: ~330 000 struct constructions a sim-second. A
py-spy profile of the served pair put 49 % of the physics thread there
against 13 % in `mj_step`. Every scan reads `data.contact.geom`, the
(ncon, 2) int view, once (`tests/test_contact_reads.py`).
"""

from __future__ import annotations

import numpy as np


def contact_pairs(data) -> np.ndarray:
  """The active contacts' geom pairs, an (ncon, 2) int array."""
  return data.contact.geom[:data.ncon]
