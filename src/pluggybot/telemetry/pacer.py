"""Real-time pacer: hold headless sim time to wall time (webserver v1).

Headless runs free-run (room_hub steps ~2.9x real time with EGL); a live
world must advance at 1x so browser viewers see the robot's actual speed.
The viewer path already paces this way (HubMission._sync sleeps off the
sim's lead); this is the same pattern as a step hook, so headless server
runs pace without a viewer.

One deliberate asymmetry: the pacer only ever SLEEPS. When the sim falls
behind wall time (an osmesa tag render burst, a swap at 1 ms timesteps) it
does nothing -- the sim catches up by simply not sleeping once it is fast
again. Sim time therefore never jumps and never stalls; lag shows up as
transient drift, which stats() reports so the accuracy is a measured
number rather than a hope.
"""

import time

PACE_PERIOD = 0.02   # sim-seconds between pacing checks (the viewer's cadence)


class RealTimePacer:
  """A step hook that sleeps the physics loop to `rate` x real time.

  Append `step_hook` to HubMission.step_hooks. rate=1.0 is a live world;
  2.0 runs sim seconds twice as fast as wall seconds. The check is
  decimated to PACE_PERIOD of sim time, so its fast path is one float
  comparison -- same discipline as the telemetry hook.
  """

  def __init__(self, data, rate: float = 1.0) -> None:
    if rate <= 0:
      raise ValueError("rate must be positive; omit the pacer to free-run")
    self.data, self.rate = data, rate
    self._next_check = 0.0
    self._wall0: float | None = None    # first hook call starts the clock
    self._t0 = 0.0
    self.max_lag = 0.0                  # worst wall-seconds behind schedule
    self.slept = 0.0                    # total wall-seconds slept

  def step_hook(self) -> None:
    t = float(self.data.time)
    if self._wall0 is None:
      self._wall0, self._t0 = time.monotonic(), t
    if t < self._next_check:
      return
    self._next_check = t + PACE_PERIOD
    ahead = (t - self._t0) / self.rate - (time.monotonic() - self._wall0)
    if ahead > 0:
      time.sleep(ahead)
      self.slept += ahead
    elif -ahead > self.max_lag:
      self.max_lag = -ahead

  def stats(self) -> dict:
    """Pacing accuracy so far: how much sim ran, how long it took, and the
    drift between the two. `drift_s` is wall-seconds ahead (+) or behind
    (-) of perfect pacing at close; `max_lag_s` the worst transient."""
    if self._wall0 is None:
      return {"sim_s": 0.0, "wall_s": 0.0, "rate": self.rate,
              "multiple": None, "drift_s": 0.0, "max_lag_s": 0.0}
    sim = float(self.data.time) - self._t0
    wall = time.monotonic() - self._wall0
    return {
      "sim_s": round(sim, 2),
      "wall_s": round(wall, 2),
      "rate": self.rate,
      "multiple": round(sim / wall, 3) if wall > 0 else None,
      "drift_s": round(sim / self.rate - wall, 3),
      "max_lag_s": round(self.max_lag, 3),
    }
