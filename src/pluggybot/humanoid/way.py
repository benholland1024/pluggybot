"""Making way, on two legs (issue #559; the quadruped's is #415's,
`legs/way.py`): a humanoid RESTING across the other robot's way is asked to
step off it, and does. A mixin over `MakeWay`, whose rule and walk are both
bodies': this says only what the humanoid's body says, in `MakeWay`'s hooks.
`HumanoidMission` is the rest of the body.

What differs is what resting is: a quadruped lies down by reflex 8.6 s after
its last motion command (`legs.posture.T_REST_S`), and a humanoid rests
STANDING -- it has had no motion command for as long, no walk of its own
waiting, no hand out and its legs on the floor (`resting`). Without it two
standing humanoids waited on each other at a doorway for 500 s of a minded
day, each walk of each giving up 1.2 m from the other (SimNotes, "The
humanoid pair served").

Its legs go to the walking policy there and then where they stand on the
standing one's (`HumanoidMission._before_step`), once its posture is the
stand's, as `to_walk_routine` hands them over; a routine that takes the
legs for the standing policy ends the step aside (`to_stand_routine`).
"""

from __future__ import annotations

from pluggybot.legs.posture import T_REST_S
from pluggybot.legs.way import STILL, MakeWay

#: How long it has had no motion command before it rests, s: the
#: quadruped's reflex's (`T_REST_S`), so a pair rests and is asked aside on
#: one clock whatever its bodies.
REST_S = T_REST_S


class HumanoidWay(MakeWay):
  """Making way (the module docstring), on `HumanoidMission`."""

  #: It rests standing (`resting`).
  RESTS_LYING = False
  #: No step aside under way, and when it last moved, sim s (`_init_way`).
  _aside = None
  last_motion_t = 0.0

  def _init_way(self) -> None:
    super()._init_way()
    #: When it was last commanded to move, sim s: what `resting` reads.
    self.last_motion_t = 0.0

  def _hand_out(self) -> bool:
    return any(p != "rest" for p in self.reach.phase.values())

  @property
  def resting(self) -> bool:
    """Resting STANDING: no motion command for `REST_S`, no walk of its own,
    no hand out and on its feet."""
    return (self.legs != "getup" and not self.driving and self._aside is None
            and not self._hand_out()
            and float(self.data.time) - self.last_motion_t >= REST_S)

  def _way_refusal(self) -> str:
    return ("getting up" if self.legs == "getup"
            else "working" if self._hand_out()
            else "walking" if self.driving
            else "standing" if not self.resting else "")

  def _aside_ready(self, command):
    """The routine's own command once it moves, the end of the step aside
    on a fall or a hand out, else None: it walks from where it stands."""
    if any(command):
      return self._end_aside("its own walk", command)
    if self.legs == "getup":
      return self._end_aside("fell", command)
    if self._hand_out():
      return self._end_aside("arm out", STILL)
    return None

  def _aside_blocked(self, t: float):
    """⚠ THE FRONT STOP'S REFLEX, as a walk's (`Navigator.drive_to_routine`):
    straight back, then a new plan. Ended on it, as the quadruped's is, a
    humanoid resting in a doorway -- its stop re-armed by every scan --
    never stepped aside at all, 3 doors of 6 (`make_way_spike.py`)."""
    if t < self.backoff_until:
      from pluggybot.humanoid.mission import command_for
      self._aside["waypoints"] = None
      return command_for(-self.BACKOFF_V, 0.0, 0.0)
    return None

  def _aside_twist(self, v: float, w: float):
    from pluggybot.humanoid.mission import command_for
    return command_for(v, 0.0, w)
