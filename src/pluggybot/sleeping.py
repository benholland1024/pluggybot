"""#775's measurement (never merged): MuJoCo 3.10's island sleeping in a
world, the robots never asleep. Off unless `PLUGGY_SLEEP` is set: "1" sleeps
at MuJoCo's own tolerance (`DEFAULT_TOLERANCE`), a number at that tolerance.

`apply(spec)` turns the flag on in a world's spec before it compiles: ⚠ the
flag must be on when the `MjData` is made, or the static bodies' poses are
never computed (MuJoCo's Simulation chapter, "Static bodies"). Every robot's
root is set `never`; everything else keeps the compiler's policy (a tree an
actuator moves never sleeps: the stations' plugs, on their balancers)."""

from __future__ import annotations

import os

ENV = "PLUGGY_SLEEP"
#: MuJoCo's own `opt.sleep_tolerance`, m/s: a velocity scaled by its DOF's
#: length, under which a tree counts toward sleep for `mjMINAWAKE` (10) steps.
DEFAULT_TOLERANCE = 1e-4


def tolerance() -> float | None:
  """The tolerance `PLUGGY_SLEEP` asks, or None: sleeping off."""
  v = (os.environ.get(ENV) or "").strip().lower()
  if v in ("", "0", "off", "no", "false"):
    return None
  if v in ("1", "on", "yes", "true"):
    return DEFAULT_TOLERANCE
  return float(v)


def robot_roots(spec) -> list:
  """The robots' root bodies in a world's spec, by name: `pluggybot`, and a
  second robot's `<prefix>pluggybot`. (⚠ `MjsBody.parent` of the world body
  segfaults MuJoCo 3.10's bindings: never walked.)"""
  from pluggybot.telemetry.protocol import ROBOT_ROOT
  return [b for b in spec.bodies if b.name == ROBOT_ROOT or b.name.endswith("_" + ROBOT_ROOT)]


def apply(spec) -> None:
  """Sleeping on in `spec` where `PLUGGY_SLEEP` asks: the enable flag, the
  tolerance, and every robot's tree `never`."""
  tol = tolerance()
  if tol is None:
    return
  import mujoco
  spec.option.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
  spec.option.sleep_tolerance = float(tol)
  for body in robot_roots(spec):
    body.sleep = mujoco.mjtSleepPolicy.mjSLEEP_NEVER
