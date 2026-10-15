"""What counts as a number where a model writes one: a procedure's literal,
a tool's spec, a scene document (#500).

Stdlib only, so the imagination's fence admits it.
"""

import math


def finite(x) -> bool:
  """A JSON number a float holds. Never a bool, and never an int past a
  float's range: JSON keeps every digit, and `math.isfinite` and `float()`
  raise OverflowError on one -- out of a define on the physics thread, and
  the served process ended with it."""
  if not isinstance(x, (int, float)) or isinstance(x, bool):
    return False
  try:
    return math.isfinite(float(x))
  except OverflowError:
    return False
