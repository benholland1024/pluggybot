"""What the robots read of a pull and a reach (#744, #746): why a `pull`
stopped -- `pull.reach` where it could reach no further, `pull.held` only
where what it pulls held the hand, `pull.joint_end` where one of its arm's own
joints read at the end of its range as it stopped -- and where the feet
went while a `reach` or a `withdraw` ran (`reach.*`, `withdraw.*`), in the
stance frame they began in, as a `step`'s are. Each rule pinned on the stub
body or a bare `Reach` (docs/Testing.md); X5's flights measured them
(docs/Observatory.md, the period of #744 and #746)."""

import math

import numpy as np
import pytest

from pluggybot.body import StubBody
from pluggybot.humanoid import arms as ha
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import reach as rc
from pluggybot.procedure import axes
from pluggybot.procedure import steps as st

from test_force_move import DT, H, _pulling
from test_guard import _floor_body
from test_hands import _bare, _fly


class _PullEnds(StubBody):
  """A stub whose pull ends the way it is told to, as `Body.pull_routine`
  reports it: its why, and its arm's joints its drivers read on their stop
  (`stops`)."""

  ends = ("distance", ())

  def pull_routine(self, side, direction, limit_n, distance_m, speed=0.06, patience=30.0,
                   give=True, verb=False):
    why, stops = self.ends.pop(0) if isinstance(self.ends, list) else self.ends
    return {"why": why, "seconds": 0.5, "moved": 0.02, "along": 0.02, "dx": 0.0,
            "dy": 0.0, "dz": 0.02, "direction": [0.0, 0.0, 1.0], "force": float(limit_n),
            "stops": list(stops)}
    yield


READ_PULL = ('    h = read("pull.held")\n    r = read("pull.reach")\n'
             '    s = read("pull.joint_end")\n')


@pytest.mark.parametrize("why, stops, held, reach, stop", [
  ("held fast", (), 1.0, 0.0, 0.0),
  # ...held at its own wrist's stop (#744: 144 of X5's 159 held-fast lifts;
  # Luca took two running for the lid's open stop)
  ("held fast", ("wrist_roll",), 0.0, 0.0, 1.0),
  # ...out of reach (Rowan asked which it was: budget or reach)
  ("out of reach", (), 0.0, 1.0, 0.0),
  ("out of reach", ("wrist_roll", "shoulder_yaw"), 0.0, 1.0, 1.0),
  # ...a joint at its end is a fact of the arm, whatever stopped it
  ("distance", ("wrist_roll",), 0.0, 0.0, 1.0),
  ("distance", (), 0.0, 0.0, 0.0),
  ("let go", (), 0.0, 0.0, 0.0),
  ("feet moved", (), 0.0, 0.0, 0.0),
  ("until", ("elbow",), 0.0, 0.0, 1.0)])
def test_a_pull_reads_why_it_stopped_and_its_own_arms_stop_apart(why, stops, held, reach, stop):
  """#744: a pull out of reach read like one gone its distance but for
  `pull.moved`, and one held at the robot's own wrist stop read `pull.held`
  1, as a lid at its open stop does. `pull.reach` reads 1 where it could
  reach no further; `pull.held` 1 only where it was held fast with none of
  its arm's joints on its stop -- what it pulls held it; `pull.joint_end` 1
  where one was, whatever stopped it. Every such pull still reads ok
  (#650)."""
  body = _PullEnds(*_bare())
  body.ends = (why, stops)
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n    p = 0\n'
             '    if pull("left", 0, 0, 1, 10, 0.1):\n        p = 1\n' + READ_PULL, body)
  assert ran["ok"] and ran["locals"]["p"] == 1, ran
  assert (ran["locals"]["h"], ran["locals"]["r"], ran["locals"]["s"]) == (held, reach, stop), ran


def test_a_pull_that_reads_false_leaves_none_of_its_new_readings():
  """As every reading of a pull: after one that read false, the last pull's
  would read as this one's (#525, #686)."""
  body = _PullEnds(*_bare())
  body.ends = [("held fast", ("wrist_roll",)), ("not out", ())]
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n    pull("left", 0, 0, 1, 10, 0.1)\n'
             '    was = read("pull.joint_end")\n'
             '    if not pull("left", 0, 0, 1, 10, 0.1):\n        p = 0\n'
             '    s = read("pull.joint_end")\n', body)
  assert ran["locals"]["was"] == 1.0, ran
  assert ran["steps"][-1]["reason"] == "pull.joint_end has no reading: your last `pull` failed", ran
  ran = _fly('def a():\n    wait(0)\n    r = read("pull.reach")\n')
  assert ran["steps"][-1]["reason"] == "pull.reach has no reading: `pull` has not run", ran


def _held_fast(r, way):
  """`r`'s pull pushed at its whole force, its hand still, until it ends."""
  r.force["left"] = 9.0 * way
  for _ in range(int(rc.FAST_S / DT) + 2):
    if r.pulls["left"] is None:
      break
    r._pulled("left", DT, H, np.eye(3))
  assert r.report["left"]["why"] == "held fast", r.report["left"]
  return r.report["left"]


def test_a_pull_ends_saying_which_of_its_arms_joints_its_drivers_read_on_their_stop():
  """#744: the body knows which: its wrist's encoder reads the roll at its
  range's end. A pull's end names its own arm's joints its drivers last
  read (`Reach.sensed`, never read afresh: a read draws noise) within
  `reach.JOINT_END_RAD` of an end of the range the robot's model of itself
  gives, or past it -- its arm's solve keeps every joint `arms.RANGE_MARGIN`
  inside, so one read there was pushed onto its stop -- and never one
  further in, as the whole body's force reading's wider `arms.STOP_RAD`
  would have (#744's review: a shoulder's yaw 19 mrad short of its end,
  pressing nothing). The other arm's are not this pull's, and a joint with
  no reading yet is at no end."""
  way = np.array([0.0, 0.0, 1.0])
  r = _pulling(way)
  sm = r.self_model
  arm, other = sm.arm["left"], sm.arm["right"]
  lo, hi = sm.ranges[arm, 0], sm.ranges[arm, 1]
  roll, elbow = hm.ARM.index("wrist_roll"), hm.ARM.index("elbow")

  def ends(at: dict):
    r_ = _pulling(way)
    r_.sensed = np.full(len(hm.JOINT_NAMES), np.nan)
    r_.sensed[arm] = (lo + hi) / 2
    r_.sensed[other] = sm.ranges[other, 0]           # the other arm on its stops
    for k, q in at.items():
      r_.sensed[arm[k]] = q
    return _held_fast(r_, way)["stops"]
  assert ends({roll: hi[roll] - 0.5 * rc.JOINT_END_RAD}) == ["wrist_roll"]
  assert ends({roll: hi[roll] + 0.01}) == ["wrist_roll"]   # pressed past it
  assert ends({roll: hi[roll] - 1.5 * rc.JOINT_END_RAD}) == []
  assert ends({roll: hi[roll] - 0.5 * ha.STOP_RAD}) == []  # near, pressing nothing
  assert ends({elbow: lo[elbow] + 0.5 * rc.JOINT_END_RAD,
               roll: lo[roll] + 0.2 * rc.JOINT_END_RAD}) == ["elbow", "wrist_roll"]
  assert ends({}) == []
  r = _pulling(way)
  assert np.isnan(r.sensed).all() and _held_fast(r, way)["stops"] == []
  assert rc.JOINT_END_RAD < ha.STOP_RAD < ha.RANGE_MARGIN


def test_the_humanoids_pull_reports_the_stops_its_reach_read_as_it_ended():
  """Through `Body`: the humanoid's pull routine says the joints its reach
  read on their stop as the pull ended (`stops`), whatever ended it."""
  body = _floor_body()

  class Grip:
    def held(self):
      return True

  class Pulling:
    def __init__(self):
      self.phase = {s: "hold" for s in hm.SIDES}
      self.report = {s: None for s in hm.SIDES}
      self.grippers = {s: Grip() for s in hm.SIDES}
      self.kept_by = {s: None for s in hm.SIDES}

    def out(self, side):
      return True

    def grip_now(self, side):
      return np.array([0.3, 0.2, 1.0]), np.eye(3)

    def pull(self, side, way, force, distance, speed, give, verb=False):
      self.phase[side] = "carry"
      return True

    def step(self):
      for s in hm.SIDES:
        if self.phase[s] == "carry":
          self.phase[s] = "hold"
          self.report[s] = {"why": "held fast", "seconds": 0.1, "moved": 0.0, "dx": 0.0,
                            "dy": 0.0, "dz": 0.0, "went": 0.0, "force": 9.0,
                            "way": [0.0, 0.0, 1.0], "stops": ["wrist_roll"]}

    def held_pitch(self):
      return 0.0
  body.mission.reach = Pulling()
  rep = body.run(body.pull_routine("left", (0.0, 0.0, 1.0), 10.0, 0.2))
  assert rep["why"] == "held fast" and rep["stops"] == ["wrist_roll"], rep


class _Swivels(StubBody):
  """A stub whose reach and withdraw move its feet as a posture swivelled
  Luca's 15-26 deg (#746): (ahead, left, m, turn, rad) in the stance frame
  it stood in as the verb began."""

  moves = {"reach": (0.0, 0.0, 0.0), "withdraw": (0.0, 0.0, 0.0)}

  def _swivel(self, verb):
    dx, dy, turn = self.moves[verb]
    c, s = math.cos(self.theta), math.sin(self.theta)
    self.x += c * dx - s * dy
    self.y += s * dx + c * dy
    self.theta += turn

  def reach_routine(self, side, pos, rot=None, **kw):
    ok = yield from super().reach_routine(side, pos, rot, **kw)
    self._swivel("reach")
    return ok

  def release_routine(self, side, refuse=False):
    why = yield from super().release_routine(side, refuse)
    self._swivel("withdraw")
    return why


def _swivelling(reach=(0.0, 0.0, 0.0), withdraw=(0.0, 0.0, 0.0)):
  body = _Swivels(*_bare())
  body.moves = {"reach": reach, "withdraw": withdraw}
  body.start_at(2.0, 1.0, math.pi / 2)
  return body


READ_REACH, READ_WITHDRAW = (
  "".join(f'    {v[0]}{k} = read("{v}.{k}")\n' for k in ("dx", "dy", "turn"))
  for v in ("reach", "withdraw"))


def test_a_reach_and_a_withdraw_report_where_the_feet_went_as_a_step_does():
  """#746: a reach can turn the feet 15-26 deg, the belief turns with them,
  and nothing the robot read said so: Luca's grips aimed where the knob had
  been. Where the stance went while each ran rides `reach.*` and
  `withdraw.*`, in the stance frame it stood in as it began, as `step.*`
  does -- so `step`'s own formula turns a place read before into the frame
  after, the world point staying where it was."""
  body = _swivelling(reach=(0.03, -0.02, 0.35), withdraw=(-0.01, 0.04, -0.1))
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n    withdraw("left")\n' + READ_REACH
             + READ_WITHDRAW,
             body)
  got = ran["locals"]
  assert ran["ok"], ran
  assert (got["rdx"], got["rdy"], got["rturn"]) == pytest.approx((0.03, -0.02, 0.35), abs=1e-9)
  assert (got["wdx"], got["wdy"], got["wturn"]) == pytest.approx((-0.01, 0.04, -0.1), abs=1e-9)
  # ...a place (0.4, 0.15) read before the reach, turned by step's formula
  body = _swivelling(reach=(0.03, -0.02, 0.35))
  bx, by, byaw = body.stance
  wx = bx + math.cos(byaw) * 0.4 - math.sin(byaw) * 0.15
  wy = by + math.sin(byaw) * 0.4 + math.cos(byaw) * 0.15
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n' + READ_REACH, body)
  dx, dy, t = (ran["locals"][k] for k in ("rdx", "rdy", "rturn"))
  c, s = math.cos(t), math.sin(t)
  x, y = c * (0.4 - dx) + s * (0.15 - dy), c * (0.15 - dy) - s * (0.4 - dx)
  bx, by, byaw = body.stance
  assert (bx + math.cos(byaw) * x - math.sin(byaw) * y,
          by + math.sin(byaw) * x + math.cos(byaw) * y) == pytest.approx((wx, wy), abs=1e-9)


class _Corrected(_Swivels):
  """A `_Swivels` whose belief a scan or a station corrected before the verb
  (`corrected`: x, y, heading), and whose heading reads, as the humanoid's
  does, in (-pi, pi]."""

  corrected = (0.0, 0.0, 0.0)

  def _swivel(self, verb):
    super()._swivel(verb)
    self.theta = math.remainder(self.theta, 2 * math.pi)

  @property
  def reckoned(self):
    x, y, yaw = self.stance
    cx, cy, cyaw = self.corrected
    return x - cx, y - cy, yaw - cyaw


def test_the_feet_are_read_in_the_frame_the_robot_stood_in_whatever_corrected_it():
  """A correction before the verb offsets the legs' reckoning from the
  belief (`Body.reckoned`), never the frame a reading is said in: the stance
  frame the robot stood in as the verb began, its believed heading -- and a
  turn across the heading's wrap reads as the turn it was."""
  body = _Corrected(*_bare())
  body.moves = {"reach": (0.05, 0.02, 0.35), "withdraw": (0.0, 0.0, 0.0)}
  body.corrected = (0.3, -0.2, 0.8)
  body.start_at(2.0, 1.0, math.pi - 0.1)
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n' + READ_REACH, body)
  got = ran["locals"]
  assert (got["rdx"], got["rdy"], got["rturn"]) == pytest.approx((0.05, 0.02, 0.35), abs=1e-9)
  assert body.theta == pytest.approx(math.pi - 0.1 + 0.35 - 2 * math.pi)


def test_a_reach_that_read_false_still_says_where_the_feet_went():
  """A reach stopped short moved the feet most (X5: Luca's chest, a median
  13 cm of the work's place across one): what it reads is the feet's,
  however the reach ended."""
  body = _swivelling(reach=(0.0, 0.01, -0.2))
  body.out_of_reach.add("left")
  ran = _fly('def a():\n    if not reach("left", 0.3, 0.2, 0.9):\n'
             '        t = read("reach.turn")\n        y = read("reach.dy")\n', body)
  assert ran["locals"]["t"] == pytest.approx(-0.2) and ran["locals"]["y"] == pytest.approx(0.01)


def test_a_reach_refused_before_it_ran_leaves_no_reading_of_the_feet():
  """A call refused before it ran leaves its verb no reading (#686): the
  last reach's would read as this one's."""
  body = _swivelling(reach=(0.05, 0.0, 0.1))
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n    was = read("reach.turn")\n'
             '    if not reach("left", 0.3, 0.2, 0.9, fx=1):\n        p = 0\n'
             '    t = read("reach.turn")\n', body)
  assert ran["locals"]["was"] == pytest.approx(0.1), ran
  assert ran["steps"][-1]["reason"] == "reach.turn has no reading: your last `reach` failed", ran
  ran = _fly('def a():\n    wait(0)\n    t = read("withdraw.dx")\n')
  assert ran["steps"][-1]["reason"] == "withdraw.dx has no reading: `withdraw` has not run", ran


def _bow(body, dp):
  """The pelvis carried `dp` (stance frame, m) over planted feet, as a
  posture carries it and the odometry follows: its model of itself posed
  so, the believed pelvis moved the same way, turned by the believed
  stance's heading."""
  odo = body.mission.odo
  syaw = body.stance[2]
  odo.sm.d.qpos[0:2] += dp
  odo.x += math.cos(syaw) * dp[0] - math.sin(syaw) * dp[1]
  odo.y += math.sin(syaw) * dp[0] + math.cos(syaw) * dp[1]


def test_the_humanoids_feet_are_reckoned_by_its_legs_never_by_a_correction():
  """A scan match or a station's look moves the belief, and with it
  `stance`: no move of the feet. `reckoned` is the stance by the legs' own
  reckoning -- the stance less every correction from outside it, summed --
  so a correction's place and turn move it not at all, and a posture
  carrying the pelvis over planted feet moves it not at all, however large
  the corrections summed before (#744's second review: the pelvis's place
  turned by the summed turn read a 16 cm bow, 90 deg of turn corrected
  since boot, as the feet moving 0.23 m). A turn corrected meanwhile leaves
  only that place turned by it."""
  body = _floor_body()
  s0, r0 = body.stance, body.reckoned
  assert s0 == r0
  x, y, yaw = body.mission.pose
  body.mission.odo.correct(x + 0.12, y - 0.07, yaw + math.pi / 2)
  s1, r1 = body.stance, body.reckoned
  assert math.hypot(s1[0] - s0[0], s1[1] - s0[1]) > 0.1 and s1[2] - s0[2] == pytest.approx(math.pi / 2)
  assert r1 == pytest.approx(r0, abs=1e-9)
  # ...and a bow after it: the stance stays where the feet are, and so does
  # the reckoning
  _bow(body, np.array([0.16, 0.0]))
  assert body.stance == pytest.approx(s1, abs=1e-9) and body.reckoned == pytest.approx(r0, abs=1e-9)
  # ...a turn corrected with the pelvis over the feet that far: at most
  # that place times the turn
  x, y, yaw = body.mission.pose
  body.mission.odo.correct(x, y, yaw + 0.05)
  r2 = body.reckoned
  assert r2[2] == pytest.approx(r0[2], abs=1e-12)
  assert math.hypot(r2[0] - r0[0], r2[1] - r0[1]) <= 0.17 * 0.05 + 1e-9


def test_the_robot_is_told_its_new_readings():
  """Every reading a served robot and a practicing mind may read is in its
  facts, with its words (`axes.SENSORS`)."""
  facts = st.hand_facts()
  for name in ("pull.reach", "pull.joint_end", "reach.dx", "reach.dy", "reach.turn",
               "withdraw.dx", "withdraw.dy", "withdraw.turn", "move_hand.short"):
    assert name in facts.sensors and name in st.served_hand_facts().sensors
    assert axes.SENSORS[name].doc
  assert set(axes.PULL_SENSORS) >= {"pull.held", "pull.reach", "pull.joint_end"}


class _MoveEnds(StubBody):
  """A stub whose carry ends the way it is told to, as `Body.move_hand_routine`
  reports it: its why, and how far short of where it was going its hand
  ended by its own reading (`shortM`; none where it never moved on)."""

  ends = ("arrived", 0.004)

  def move_hand_routine(self, side, pos, stiffness=1.0, speed=0.06, patience=20.0,
                        path=None, turns=None, verb=False):
    why, short = self.ends.pop(0) if isinstance(self.ends, list) else self.ends
    rep = {"why": why, "seconds": 1.5, "moved": 0.1, "dx": 0.0, "dy": 0.0, "dz": 0.1}
    return rep if short is None else {**rep, "shortM": short}
    yield


def test_a_carry_held_back_says_how_far_short_by_its_own_reading_and_names_no_cause():
  """#774 (b): a carry still more than 1 cm short of its goal 1 s after its
  command got there read "held back short of it: what it holds or pushes on
  stopped it" -- in C2d's flights a load sagged the hand short, and an empty
  one truly stood inside the 1 cm, nothing stopping either. It says how far
  short its hand ended by its own reading, naming no cause, and
  `move_hand.short` reads it after any end -- none after a hand not out, as
  every reading of a verb that read false before it moved."""
  body = _MoveEnds(*_bare())
  body.ends = [("held back", 0.021), ("arrived", 0.004), ("not out", None)]
  ran = _fly('def a():\n    reach("right", 0.3, -0.2, 0.9)\n'
             '    if not move_hand("right", 0.35, -0.2, 0.9):\n        s = read("move_hand.short")\n'
             '    move_hand("right", 0.35, -0.2, 0.92)\n    t = read("move_hand.short")\n'
             '    if not move_hand("right", 0.35, -0.2, 0.95):\n        u = 0\n'
             '    v = read("move_hand.short")\n', body)
  held = ran["steps"][1]
  assert not held["ok"] and held["reason"] == (
    "your right hand stopped 2.1 cm short of it, by its own reading"), held
  assert ran["locals"]["s"] == pytest.approx(0.021), ran
  assert ran["locals"]["t"] == pytest.approx(0.004) and ran["locals"]["u"] == 0, ran
  assert ran["steps"][-1]["reason"] == (
    "move_hand.short has no reading: your last `move_hand` failed"), ran
  assert not any("holds or pushes" in w for w in st.MOVE_WHY.values())
  # ...nor before any `move_hand` ran
  ran = _fly('def a():\n    wait(0)\n    r = read("move_hand.short")\n')
  assert ran["steps"][-1]["reason"] == "move_hand.short has no reading: `move_hand` has not run"
