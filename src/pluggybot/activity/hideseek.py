"""Hide and seek: the first two-role game (issue #167, M12; #120's example;
on legs, #404).

An ACTIVITY (docs/ActivityPattern.md), because a game is a mechanism that
owns world state: sensed every physics step off both robots' bodies and
rays between them, latched, and shipped as flags. The two robots each run
a role's steps (`lifecycle.hide_and_seek_program`: the hider's `hide`, the
seeker's `seek`); this is the referee, and it reads the WORLD, never either
robot's account:

  hiding    the hider is on its way; the seeker counts where it stands
  seeking   from `SEEK_HEAD_START_S` after the game starts, the seeker
            searches; every step the distance between the two torsos is
            measured, and within `FIND_WITHIN_M`, every `LOS_EVERY_S`, rays
            from the seeker's LIDAR to every geom of the hider's body
  found     latched the first time the seeker is within `FIND_WITHIN_M`
            WITH line of sight to any part of the hider -- a wall or a couch
            between them is not a find; a leg round a corner is
  over      found, or `SEEK_S` of seeking without one: the hider wins

No new sensing: the seeker does not have to KNOW it found the hider (that
is a later, harder game); the criterion is what the world can measure, as
#120 decided for every challenge. One verdict for both robots
(`scoring.eval_hide_and_seek`), banked on the winner's wallet by the pair.

THE EYE IS THE SEEKER'S LIDAR, on the quadruped's rear mast (0.51 m up),
and its own body is no wall: a ray that meets the seeker's geoms first is
cast on from past them. MEASURED, to a robot lying 0.7-1.0 m in front of
it every one of the 41 rays met the seeker's own body first -- blind
straight ahead -- and to one standing 1.4 m off, 29 of them, the rover's
one ray at the torso among them.

ONE REFEREE A WORLD, GAME AFTER GAME (#404): built idle with the world's
activities, given each game's roles as its last role is claimed (`assign`,
which ends the last game's record), and STARTED ONCE BOTH ROLES' ERRANDS
HAVE BEGUN (`begin`): the robot that took the first role was free until
the other took the last, and is often still busy, so a clock started by
one role alone was a game played against a robot charging or dead. Called
off -- nobody wins -- when both have not begun `START_WITHIN_S` after the
roles were taken, or when a player dies (`call_off`, the pair's watch). A
game does not outlive its process: a restart's referee is idle (the board
puts the job back on offer, TaskPattern §1).
"""

import math

import mujoco
import numpy as np

from pluggybot.activity.base import Activity
from pluggybot.perception.lidar import robot_geoms
from pluggybot.robot import RobotHandle

#: The seeker counts this long before searching -- the head start: at the
#: drive's 0.4 m/s cruise, less its turns, about 6 m of walk to a spot.
#: MEASURED on legs (`solve.py --feature hide_and_seek`, 14 games): every
#: hider out of sight at its spot in 10.3-20.5 s, median 17 s.
SEEK_HEAD_START_S = 20.0
#: ...and searches this long before the hider wins. MEASURED on legs, the
#: same 14 games: the seeker found 6 at 82-249 s; at the rover's 120 s, 1
#: of 11 -- an honest search of its own map outward covers less than a
#: sweep of surveyed spots did.
SEEK_S = 240.0
#: A find: this close, torso to torso, with line of sight. Two body
#: lengths, as the rover's 1.0 m was: the quadruped standing reaches 0.38 m
#: from its torso's centre, so two of them nose to tail are 0.76 m.
FIND_WITHIN_M = 1.5
#: The rays are cast this often while the two are within reach, s: MEASURED,
#: 41 rays and their casts past the seeker's own body cost 0.15-0.35 ms a
#: check (up to 1.1 ms on a loaded box) -- a tenth of a physics step's
#: worth, spread over the 50 between -- and a find 0.1 s late is 4 cm.
LOS_EVERY_S = 0.1
#: How many times one ray is cast on past the seeker's own geoms: a ray cast
#: from inside a geom meets where it leaves, so each geom it crosses costs
#: two, and from the mast a ray crosses the arm and a leg at most.
PAST_OWN_BODY = 8
#: Roles taken and both roles' errands not begun this long after: the game
#: is called off, nobody paid -- a claimed job would otherwise hold its
#: offer's target, and no game could be offered again.
START_WITHIN_S = 600.0


class HideAndSeek(Activity):
  name = "hide_and_seek"

  def __init__(self, model, hider: RobotHandle | None = None,
               seeker: RobotHandle | None = None,
               head_start_s: float = SEEK_HEAD_START_S, seek_s: float = SEEK_S,
               find_within_m: float = FIND_WITHIN_M,
               start_within_s: float = START_WITHIN_S) -> None:
    super().__init__()
    self.model = model
    self.head_start_s, self.seek_s = head_start_s, seek_s
    self.find_within_m = find_within_m
    self.start_within_s = start_within_s
    self.on_over: list = []
    self._reset()
    # The roles are known only once both are CLAIMED, but the referee has to
    # be in the world's activity set from the moment the game can be offered
    # -- the stream's header lists activities once, and a referee added later
    # would ship flags under a name no consumer was told about. So it is
    # built unassigned (`idle`) and `assign`ed at the claim.
    if hider is not None and seeker is not None:
      self.assign(hider, seeker)

  def _reset(self) -> None:
    self.hider: RobotHandle | None = None
    self.seeker: RobotHandle | None = None
    self.task_id = ""
    self.hider_bid = self.seeker_bid = self.seeker_eye = -1
    self.hider_gids = np.zeros(0, dtype=np.int32)
    self._seeker_gids: set = set()
    self._hider_set: set = set()
    self.assigned_at: float | None = None
    self.started_at: float | None = None
    self.found_at: float | None = None
    self.over_at: float | None = None
    #: the roles whose errands have begun, and why the game was called off
    self.begun: set = set()
    self.called_off = ""
    self._los, self._next_los = False, -math.inf
    self.set(phase="idle", distanceM=None, los=False, foundAtS=None,
             overAtS=None, winner="")

  def assign(self, hider: RobotHandle, seeker: RobotHandle, task_id: str = "",
             t: float | None = None) -> None:
    """Who hides and who seeks in the game for `task_id` -- both roles
    claimed. The last game's record ends here, and not when it is called,
    so its verdict stays on the wire until the next game begins."""
    self._reset()
    self.hider, self.seeker, self.task_id = hider, seeker, task_id
    self.assigned_at = None if t is None else float(t)
    self._resolve()

  def _resolve(self) -> None:
    """Every id the referee reads, by name: the two roots, the seeker's
    LIDAR site and both robots' geoms."""
    m = self.model
    self.hider_bid = m.body(self.hider.root).id
    self.seeker_bid = m.body(self.seeker.root).id
    self.seeker_eye = m.site(self.seeker.el("lidar")).id
    self.hider_gids = np.array(sorted(robot_geoms(m, self.hider.root)), dtype=np.int32)
    self._hider_set = set(self.hider_gids.tolist())
    self._seeker_gids = robot_geoms(m, self.seeker.root)

  def rebind(self, model, data) -> None:
    self.model = model
    if self.assigned:
      self._resolve()

  def restore_kept(self, state: dict) -> None:
    """A game does not outlive its process: whatever the last referee was
    doing, this one is idle (the module docstring)."""
    self._reset()

  @property
  def assigned(self) -> bool:
    return self.hider is not None and self.seeker is not None

  def playing(self, task_id: str) -> bool:
    """Is this referee's game the one for `task_id`, and not yet called?"""
    return self.assigned and self.task_id == task_id and self.over_at is None

  def over_for(self, task_id: str) -> bool:
    """Is the game for `task_id` over -- called, or no longer this
    referee's? What a role's walks stop on."""
    return not self.playing(task_id)

  @property
  def seeker_root(self) -> str:
    return self.seeker.root if self.seeker is not None else ""

  def hiding_left(self, t: float) -> float:
    """Seconds of the head start left at `t`: all of it before the game
    starts."""
    if self.started_at is None:
      return self.head_start_s
    return max(0.0, self.started_at + self.head_start_s - float(t))

  def begin(self, role: str, task_id: str, t: float) -> None:
    """A role's errand for the game `task_id` has begun
    (`HubLifecycle._run_program_routine`): the game starts once BOTH have
    (the module docstring). An errand of another game is no begin of this
    one."""
    if not self.playing(task_id) or role not in ("hider", "seeker"):
      return
    self.begun.add(role)
    if len(self.begun) == 2:
      self.start(t)

  def start(self, t: float) -> None:
    """The game's clock: the head start, then the seeking. `begin` starts
    it; a test may start it by hand."""
    if self.started_at is None and self.assigned and self.over_at is None:
      self.started_at = float(t)
      self.set(phase="hiding")

  def call_off(self, t: float, why: str) -> None:
    """End the game with nobody winning, saying why: the roles were not
    both begun in time, or a player died (the pair's watch,
    `pair.referee_games`) -- a game is played by two robots, and a verdict
    over one that stopped playing would pay for whatever it did instead."""
    if not self.assigned or self.over_at is not None:
      return
    self.over_at = float(t)
    self.called_off = why
    self._los = False
    self.set(phase="over", winner="", los=False,
             overAtS=(round(self.over_at - self.started_at, 2)
                      if self.started_at is not None else None))
    for hook in list(self.on_over):
      hook(self)

  @property
  def phase(self) -> str:
    return self.flags["phase"]

  def _line_of_sight(self, model, data) -> bool:
    """Any ray from the seeker's LIDAR to the centre of a geom of the
    hider's body that meets the hider before anything else -- the seeker's
    own geoms passed through (the module docstring)."""
    eye = np.array(data.site_xpos[self.seeker_eye], dtype=np.float64)
    vec = np.array(data.geom_xpos[self.hider_gids], dtype=np.float64) - eye
    dist = np.linalg.norm(vec, axis=1)
    if not (dist > 1e-6).all():
      return True
    dirs = vec / dist[:, None]
    n = len(dirs)
    hit = np.zeros(n, dtype=np.int32)
    far = np.zeros(n, dtype=np.float64)
    mujoco.mj_multiRay(model, data, eye, np.ascontiguousarray(dirs.reshape(-1)),
                       None, 1, self.seeker_bid, hit, far, None, n, mujoco.mjMAXVAL)
    one = np.zeros(1, dtype=np.int32)
    for i in range(n):
      g, gone = int(hit[i]), 0.0
      for _ in range(PAST_OWN_BODY):           # past the seeker's own body
        if g < 0 or g not in self._seeker_gids:
          break
        gone += float(far[i]) + 1e-4
        far[i] = mujoco.mj_ray(model, data, eye + dirs[i] * gone, dirs[i],
                               None, 1, -1, one)
        g = int(one[0])
      if g in self._hider_set:
        return True
    return False

  def sense(self, model, data) -> None:
    if not self.assigned or self.over_at is not None:
      return
    t = float(data.time)
    if self.started_at is None:
      if self.assigned_at is not None and t - self.assigned_at >= self.start_within_s:
        self.call_off(t, f"the two roles were not both begun within "
                         f"{self.start_within_s:.0f} s")
      return
    hider = data.xpos[self.hider_bid]
    seeker = data.xpos[self.seeker_bid]
    dist = float(math.hypot(hider[0] - seeker[0], hider[1] - seeker[1]))
    phase = "hiding" if t - self.started_at < self.head_start_s else "seeking"
    if phase == "seeking" and dist <= self.find_within_m:
      if t >= self._next_los:
        self._next_los = t + LOS_EVERY_S
        self._los = self._line_of_sight(model, data)
    else:
      self._los, self._next_los = False, -math.inf
    if phase == "seeking":
      if dist <= self.find_within_m and self._los:
        self.found_at = t
        self.over_at = t
        phase = "found"
      elif t - self.started_at >= self.head_start_s + self.seek_s:
        self.over_at = t
        phase = "over"
    self.set(phase=phase, distanceM=round(dist, 3), los=bool(self._los),
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
    ran to a decision at all -- a game called off was not, and why."""
    return {**self.flags,
            "played": (self.over_at is not None and self.started_at is not None
                       and not self.called_off),
            "calledOff": self.called_off,
            "seekS": self.seek_s, "findWithinM": self.find_within_m}
