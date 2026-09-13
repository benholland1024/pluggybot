"""Hide and seek: the first two-role game (issue #167, M12; #120's example).

An ACTIVITY (docs/ActivityPattern.md), because a game is a mechanism that
owns world state: sensed every physics step off both robots' poses and a
raycast between them, latched, and shipped as flags. The two robots each
run a role's steps (`HIDE_AND_SEEK` in mission/errand.py -- #58's `roles`
slot filled); this is the referee, and it reads the WORLD, never either
robot's account:

  hiding    the hider is on its way; the seeker counts (waits) at its start
  seeking   from `SEEK_HEAD_START_S` after the game starts, the seeker
            searches; every step the distance between the two chassis and a
            ray from the seeker's lidar to the hider are measured
  found     latched the first step the seeker is within `FIND_WITHIN_M`
            WITH line of sight -- a wall between them is not a find
  over      found, or `SEEK_S` of seeking without one: the hider wins

No new sensing: the seeker does not have to KNOW it found the hider (that
is a later, harder game); the criterion is what the world can measure, as
#120 decided for every challenge. One verdict for both robots
(`scoring.eval_hide_and_seek`), banked on the winner's wallet by the pair.
"""

import math

import mujoco
import numpy as np

from pluggybot.activity.base import Activity
from pluggybot.robot import RobotHandle

#: The seeker counts this long before searching -- the head start.
SEEK_HEAD_START_S = 20.0
#: ...and searches this long before the hider wins.
SEEK_S = 120.0
#: A find: this close, with line of sight. Two body lengths.
FIND_WITHIN_M = 1.0


class HideAndSeek(Activity):
  name = "hide_and_seek"

  def __init__(self, model, hider: RobotHandle | None = None,
               seeker: RobotHandle | None = None,
               head_start_s: float = SEEK_HEAD_START_S, seek_s: float = SEEK_S,
               find_within_m: float = FIND_WITHIN_M) -> None:
    super().__init__()
    self.model = model
    # The roles are known only once both are CLAIMED, but the referee has to
    # be in the world's activity set from the moment the game is offered --
    # the stream's header lists activities once, and a referee added later
    # would ship flags under a name no consumer was told about. So it is
    # built unassigned (`idle`) and `assign`ed at the claim.
    self.hider: RobotHandle | None = None
    self.seeker: RobotHandle | None = None
    self.hider_bid = self.seeker_bid = self.seeker_eye = self.hider_geom = -1
    if hider is not None and seeker is not None:
      self.assign(hider, seeker)
    self.head_start_s, self.seek_s = head_start_s, seek_s
    self.find_within_m = find_within_m
    self.started_at: float | None = None
    self.found_at: float | None = None
    self.over_at: float | None = None
    self.on_over: list = []
    self._geomid = np.zeros(1, dtype=np.int32)
    self.set(phase="idle", distanceM=None, los=False, foundAtS=None,
             overAtS=None, winner="")

  def assign(self, hider: RobotHandle, seeker: RobotHandle) -> None:
    """Who hides and who seeks -- both roles claimed."""
    self.hider, self.seeker = hider, seeker
    self.hider_bid = self.model.body(hider.root).id
    self.seeker_bid = self.model.body(seeker.root).id
    self.seeker_eye = self.model.site(seeker.el("lidar")).id
    self.hider_geom = self.model.geom(hider.el("chassis")).id

  @property
  def assigned(self) -> bool:
    return self.hider is not None and self.seeker is not None

  def start(self, t: float) -> None:
    """The game is on: both roles claimed, both errands under way."""
    if self.started_at is None and self.assigned:
      self.started_at = float(t)
      self.set(phase="hiding")

  @property
  def phase(self) -> str:
    return self.flags["phase"]

  def _line_of_sight(self, model, data, eye, target) -> bool:
    vec = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    dist = float(np.linalg.norm(vec))
    if dist < 1e-6:
      return True
    hit = mujoco.mj_ray(model, data, np.ascontiguousarray(eye),
                        np.ascontiguousarray(vec / dist), None, 1,
                        self.seeker_bid, self._geomid)
    if hit < 0.0:
      return False
    return int(model.geom_bodyid[int(self._geomid[0])]) == self.hider_bid \
        or int(model.body_rootid[int(model.geom_bodyid[int(self._geomid[0])])]) == self.hider_bid

  def sense(self, model, data) -> None:
    if self.started_at is None or self.over_at is not None:
      return
    t = float(data.time)
    hider = data.xpos[self.hider_bid]
    seeker = data.xpos[self.seeker_bid]
    dist = float(math.hypot(hider[0] - seeker[0], hider[1] - seeker[1]))
    phase = "hiding" if t - self.started_at < self.head_start_s else "seeking"
    los = False
    if phase == "seeking":
      los = self._line_of_sight(model, data, data.site_xpos[self.seeker_eye],
                                (hider[0], hider[1], hider[2] + 0.1))
      if dist <= self.find_within_m and los:
        self.found_at = t
        self.over_at = t
        phase = "found"
      elif t - self.started_at >= self.head_start_s + self.seek_s:
        self.over_at = t
        phase = "over"
    self.set(phase=phase, distanceM=round(dist, 3), los=bool(los),
             foundAtS=(round(self.found_at - self.started_at, 2)
                       if self.found_at is not None else None),
             overAtS=(round(self.over_at - self.started_at, 2)
                      if self.over_at is not None else None),
             winner=("seeker" if self.found_at is not None else
                     ("hider" if self.over_at is not None else "")))
    if self.over_at is not None:
      for hook in list(self.on_over):
        hook(self)

  def measurements(self) -> dict:
    """What the evaluator reads: the referee's flags, plus whether the game
    ran to a decision at all."""
    return {**self.flags, "played": self.over_at is not None,
            "seekS": self.seek_s, "findWithinM": self.find_within_m}
