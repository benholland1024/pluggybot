#!/usr/bin/env python
"""Where the humanoid's belief drifts, taken apart (issue #572, stage 2;
SimNotes, "Where the humanoids get lost"). One humanoid in its house -- the
served body, its station on the living room's wall -- flies the loop's own
explore, then the loop's own walks across the street to the storeroom,
where its jobs' counter is (#559's stage 3), and back to the station's
standoff, and is sent to charge: its belief against the truth throughout.
Its error is taken apart as it is made -- each tick of its reckoning, each
move of the belief from outside it -- into what put it there:

  gyro      the heading its IMU's integration adds: the humanoid's heading
            is its IMU's (`humanoid.odometry`), so its legs put none
  hold      what its reckoning took off the gyro's: the turn it held still
            at a standstill and the offset it learned there (`imu.
            HeadingHold`, #593)
  legs      the position its legs' odometry adds, each step laid through
            the TRUE heading
  heading   the position the believed heading adds: the same steps laid
            through it, less through the truth
  matcher   the scan matcher's corrections (`Navigator._match`)
  station   the station's fixes (`HumanoidBody._fix_at`)

They add up to the belief's error (the residual is said). Two passengers
ride the same walk off the same senses: the odometry alone, no matcher,
and the legs alone, through the true heading.

    MUJOCO_GL=egl uv run python scripts/humanoid_drift_spike.py
    ... --seed rowan --start hall     # Rowan's IMU and joints, from its start
    ... --explore 240 --trips 2 --rest 60 --no-charge --out run.json
    ... --idle-at 12,0 --idle 180     # first stood idle in the street
    ... --learned                     # its gyro's offset taken off its heading
    ... --table run*.json             # runs side by side, and by zone
    ... --falls [--fall-explore S]    # shoved over six ways after an explore:
                                      # the belief as it stands, and after (#572)
    ... --still boot|walk|turn|standup  # #593: standing after it, no scan matched
    ... --done [--no-wait] [--no-hold]  # #593: the pair, the second's day over first

`--seed` draws the IMU's offset and scale and the joints' noise: `luca` and
`rowan` are the served pair's own (the first robot's and the second's),
anything else a draw keyed by it. The truth is ours, read to take the
error apart; nothing here steers off it. A fall its get-up has not stood
from in `GETUP_WAIT_S` (40 s) ends the flight, and `--falls`' too, `stuck`,
the rows until then kept (`stuck_guard`).

#593's two scenes (SimNotes, "A robot left standing when its run ends, and a
heading held at a standstill"): `--still` stands one humanoid still
`--seconds` (180) after booting, a walk, a turn on the spot or a
stand-up's warp, its scan matcher off -- as when every scan is refused --
and its gyro's offset the part's worst, its heading held and its IMU's own
against the truth (`--hold-after S` moves when the hold may begin); `--done` flies the served
pair with the second robot's day over at once while the first's runs on,
past the run's `--seconds` (300) if it charges, and reads the second's
belief while it waits -- `--no-wait` held STILL by the loop, `--no-hold`
with no heading hold, both as the loop was before #593.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from station_spike import _life, zone_of  # noqa: E402  (the house as the station's trials fly it)

from pluggybot.tick import MissionAborted  # noqa: E402

#: A row of the belief against the truth this often, sim s.
EVERY_S = 5.0
#: A get-up is waited for this long at most, s (`HumanoidBody.GETUP_BUDGET_S`'s
#: margin: the slowest in the house stood at 30): `--falls`' wait after its
#: shove, and in every flight here the get-up after which it ends `stuck`
#: (`stuck_guard`).
GETUP_WAIT_S = 40.0
#: The walk across the street goes to the storeroom's middle, in front of
#: its counter (`home.storeroom`), and back to the station's standoff, each
#: walk given this long, s.
STORE_XY = (25.0, -3.0)
WALK_TIMEOUT_S = 300.0
#: The position's shares and the heading's, in the order they are said;
#: `set` is any other move of the belief (none in a flight here).
SHARES = ("legs", "heading", "matcher", "station", "set")
TURNS = ("gyro", "hold", "matcher", "station", "set")


def wrap(a: float) -> float:
  return math.remainder(a, 2 * math.pi)


def draws(body, seed: str) -> dict:
  """The IMU's and the joints' draws: a served robot's own -- `luca` the
  first robot's, `rowan` the second's -- or another's, keyed by `seed`.
  What the gyro drew, deg/s a side and its scale's error, %."""
  from pluggybot.perception.imu import Imu
  key = {"luca": "humanoid:", "rowan": "humanoid:r2_"}.get(seed, f"drift:{seed}")
  m = body.mission
  m.stand.trunk.imu = imu = Imu(f"{key}:pelvis")
  m.stand.joints.reseed(key)
  return {"gyroOffsetDegS": [round(math.degrees(v), 4) for v in imu.gyro_bias],
          "gyroScalePct": [round(100 * (v - 1), 3) for v in imu.gyro_scale]}


def learned(body) -> None:
  """The gyro's offset taken off what its heading filter integrates, as a
  zero-rate update that had learned it would -- learned exactly, at once,
  where the odometry's own learns it settled at rest (`imu.HeadingHold`,
  #593). The walking policy still reads the gyro raw: only the reckoning
  was changed."""
  trunk = body.mission.stand.trunk
  att, offset = trunk.attitude, trunk.imu.gyro_bias.copy()
  step = att.step
  att.step = lambda gyro, accel, dt: step(np.asarray(gyro) - offset, accel, dt)


def _zone() -> dict:
  return {"s": 0.0, "walkedM": 0.0, "turnedRad": 0.0, "gyro": 0.0,
          **{k: np.zeros(2) for k in SHARES}, **{f"{k}Turn": 0.0 for k in TURNS[1:]},
          "verdicts": {}, "scans": 0, "returns": 0}


class Tracker:
  """The belief against the truth, taken apart as it is made (the module
  docstring): the reckoning's every tick and every move of the belief from
  outside it, by whose it was, summed over the flight and over each zone
  the robot truly stood in."""

  def __init__(self, body, every_s: float = EVERY_S) -> None:
    self.m, self.d, self.every_s = body.mission, body.data, every_s
    self.phase, self.source = "start", "set"
    self.rows: list[dict] = []
    self.zones: dict[str, dict] = {}
    self.due, self.residual = 0.0, [0.0, 0.0]
    tx, ty, th = self.m.true_pose()
    bx, by, bh = self.m.pose
    self.e0 = (np.array([bx - tx, by - ty]), wrap(bh - th))
    self.shares = {k: np.zeros(2) for k in SHARES}
    self.turns = dict.fromkeys(TURNS, 0.0)
    self.truth, self.t = (tx, ty, th), float(self.d.time)
    self.att = self.m.odo.att
    self.att_yaw = self.att.yaw()
    #: The odometry alone's heading is its IMU's from the truth it began at.
    self.gyro_ref = th - self.att_yaw
    self.odo_only, self.legs_only = np.array([tx, ty]), np.array([tx, ty])
    self.walked = self.turned = 0.0
    _patch(self)

  def _zone(self) -> dict:
    return self.zones.setdefault(zone_of(*self.truth[:2]), _zone())

  def tick(self, odo, before) -> None:
    """One tick of the reckoning (`StanceOdometry.step`): its step, laid
    through the heading it believed, against the truth's; and what its
    hold took off the gyro's turn (`imu.HeadingHold`: its heading's
    `offset`, which nothing else moves inside a tick)."""
    bx, by, byaw, boff = before
    held = odo.offset - boff
    dx, dy = odo.x - bx, odo.y - by
    c, s = math.cos(byaw), math.sin(byaw)
    ahead, left = c * dx + s * dy, -s * dx + c * dy
    tx, ty, th = self.m.true_pose()
    px, py, ph = self.truth
    ct, st = math.cos(th), math.sin(th)
    laid = np.array([ct * ahead - st * left, st * ahead + ct * left])
    legs, heading = laid - (tx - px, ty - py), np.array([dx, dy]) - laid
    if odo.att is not self.att:          # a filter built afresh: the same heading on
      self.gyro_ref += self.att_yaw - odo.att.yaw()
      self.att, self.att_yaw = odo.att, odo.att.yaw()
    a, turned = self.att.yaw(), wrap(th - ph)
    gyro = wrap(a - self.att_yaw) - turned
    self.att_yaw = a
    self.turns["gyro"] += gyro
    self.turns["hold"] += held
    g = a + self.gyro_ref + self.turns["hold"]
    self.odo_only += (math.cos(g) * ahead - math.sin(g) * left,
                      math.sin(g) * ahead + math.cos(g) * left)
    self.legs_only += laid
    self.shares["legs"] += legs
    self.shares["heading"] += heading
    walked = math.hypot(tx - px, ty - py)
    self.walked, self.turned = self.walked + walked, self.turned + turned
    t = float(self.d.time)
    z = self._zone()
    z["s"] += t - self.t
    z["walkedM"] += walked
    z["turnedRad"] += turned
    z["gyro"] += gyro
    z["holdTurn"] += held
    z["legs"] += legs
    z["heading"] += heading
    self.truth, self.t = (tx, ty, th), t
    if t >= self.due:
      self.row()

  def moved(self, before, after) -> None:
    """The belief moved from outside the reckoning: by `source`."""
    d = np.array([after[0] - before[0], after[1] - before[1]])
    turn = wrap(after[2] - before[2])
    self.shares[self.source] += d
    self.turns[self.source] += turn
    z = self._zone()
    z[self.source] += d
    z[f"{self.source}Turn"] += turn

  def verdict(self, got, ranges) -> None:
    """A scan's match: its verdict, and how many of its returns lay within
    the matcher's range (`scan_match.MATCH_RANGE_M`) -- the walls it had to
    go on."""
    from pluggybot.mapping.scan_match import MATCH_RANGE_M
    z = self._zone()
    v = z["verdicts"]
    v[got.why] = v.get(got.why, 0) + 1
    if got.accepted and got.held:
      v["held"] = v.get("held", 0) + 1
    z["scans"] += 1
    z["returns"] += int(np.count_nonzero(ranges < MATCH_RANGE_M))

  def off(self) -> list:
    """The belief's error now, m and deg."""
    (bx, by, bh), (tx, ty, th) = self.m.pose, self.m.true_pose()
    return [round(math.hypot(bx - tx, by - ty), 3), round(math.degrees(wrap(bh - th)), 2)]

  def row(self) -> None:
    """A row at this tick, and the shares checked against the error they
    take apart."""
    t = float(self.d.time)
    self.due = (math.floor(t / self.every_s) + 1) * self.every_s
    tx, ty, th = self.truth
    bx, by, bh = self.m.pose
    err, dyaw = np.array([bx - tx, by - ty]), wrap(bh - th)
    apart = self.e0[0] + sum(self.shares.values())
    turned = self.e0[1] + sum(self.turns.values())
    self.residual = [max(self.residual[0], float(np.linalg.norm(apart - err))),
                     max(self.residual[1], abs(wrap(turned - dyaw)))]
    self.rows.append({
      "t": round(t, 1), "phase": self.phase, "zone": zone_of(tx, ty),
      "true": [round(tx, 2), round(ty, 2), round(math.degrees(th), 1)],
      "errM": round(float(np.linalg.norm(err)), 3), "errDeg": round(math.degrees(dyaw), 2),
      "share": {k: [round(float(v[0]), 3), round(float(v[1]), 3)] for k, v in self.shares.items()},
      "turn": {k: round(math.degrees(v), 2) for k, v in self.turns.items()},
      "odoM": round(float(np.linalg.norm(self.odo_only - (tx, ty))), 3),
      "legsM": round(float(np.linalg.norm(self.legs_only - (tx, ty))), 3),
      "walkedM": round(self.walked, 1), "turnedDeg": round(math.degrees(self.turned), 1),
      "counts": dict(self.m.matcher.counts)})

  def zone_table(self) -> list[dict]:
    """What each zone added: time, distance, the true turning, the shares
    (m as vectors, deg), the matcher's verdicts and its returns a scan."""
    out = []
    for name, z in self.zones.items():
      row = {"zone": name, "s": round(z["s"], 1), "walkedM": round(z["walkedM"], 2),
             "turnedDeg": round(math.degrees(z["turnedRad"]), 1),
             "gyroDeg": round(math.degrees(z["gyro"]), 3),
             "verdicts": z["verdicts"], "scans": z["scans"],
             "returnsPerScan": round(z["returns"] / max(z["scans"], 1), 1)}
      for k in SHARES:
        row[f"{k}M"] = [round(float(z[k][0]), 4), round(float(z[k][1]), 4)]
      for k in TURNS[1:]:
        row[f"{k}Deg"] = round(math.degrees(z[f"{k}Turn"]), 3)
      out.append(row)
    return out


def _patch(tr: Tracker) -> None:
  """The reckoning's ticks and the belief's moves, told to `tr`: the
  methods that make them wrapped on their classes -- in this process, which
  flies one robot."""
  from pluggybot.humanoid.body import HumanoidBody
  from pluggybot.humanoid.mission import HumanoidMission
  from pluggybot.humanoid.odometry import StanceOdometry
  step, set_pose = StanceOdometry.step, HumanoidMission._set_pose
  match, fix_at = HumanoidMission._match, HumanoidBody._fix_at

  def ticked(odo, q, gravity_b):
    before = (odo.x, odo.y, odo.yaw, odo.offset)
    step(odo, q, gravity_b)
    if odo is tr.m.odo:
      tr.tick(odo, before)

  def moved(m, x, y, theta):
    before = m.pose
    set_pose(m, x, y, theta)
    if m is tr.m:
      tr.moved(before, m.pose)

  def matched(m, angles, ranges, origin=None):
    if m is not tr.m:
      return match(m, angles, ranges, origin)
    tr.source = "matcher"
    try:
      got = match(m, angles, ranges, origin)
    finally:
      tr.source = "set"
    if got is not None:
      tr.verdict(got, ranges)
    return got

  def fixed(body, seen):
    tr.source = "station"
    try:
      return fix_at(body, seen)
    finally:
      tr.source = "set"

  StanceOdometry.step, HumanoidMission._set_pose = ticked, moved
  HumanoidMission._match, HumanoidBody._fix_at = matched, fixed


class Stuck(MissionAborted):
  """A get-up that never stood (`stuck_guard`); its one argument says when
  it fell and where it lies."""


def stuck_guard(body, wait_s: float = GETUP_WAIT_S):
  """A step hook that ends a flight once the get-up has had the legs `wait_s`
  without standing: `Stuck`. ⚠ The loop's get-up has no budget of its own:
  served, the `stuck` death's stand-up closes it, and this lifecycle is not
  mortal (SimNotes, "A flight that never got up")."""
  since: list[float | None] = [None]

  def hook() -> None:
    if body.posture != "getting_up":
      since[0] = None
      return
    t = float(body.data.time)
    if since[0] is None:
      since[0] = t
    elif t - since[0] >= wait_s:
      x, y, _ = body.true_pose()
      raise Stuck({"fellT": round(since[0], 1), "t": round(t, 1), "zone": zone_of(x, y),
                   "at": [round(x, 2), round(y, 2)]})
  return hook


def _flight():
  """The house and a lifecycle on it, as the station's trials fly it
  (`station_spike._life`), a get-up that never stands ending the flight
  (`stuck_guard`). (life, body, station)."""
  life, body, station = _life()
  body.step_hooks.append(stuck_guard(body))
  return life, body, station


def fly(seed: str = "luca", start: str = "start", explore_s: float = 240.0, trips: int = 2,
        rest_s: float = 0.0, charge: bool = True, idle_at=None, idle_s: float = 0.0,
        offset_learned: bool = False) -> dict:
  """One humanoid from `start` (`start` Luca's, `hall` Rowan's), its draws
  `seed`'s -- its gyro's offset taken off its heading if `offset_learned`
  (`learned`) -- a look round, the loop's explore for `explore_s`, a walk
  to `idle_at` to stand idle there `idle_s`, if given, `trips` walks to the
  storeroom and back to the station's standoff -- standing still `rest_s`
  at the storeroom, as a job would -- and, if `charge`, the loop's charge
  path and the station's fix as it charges. The rows, each walk's end, the
  shares over the flight and over each zone -- and `stuck`, where a fall
  it never got up from ended it (`stuck_guard`), the rows until then."""
  from pluggybot.lifecycle import HUMANOID_HOME, world_config
  cfg = world_config(HUMANOID_HOME)
  starts = {"start": cfg["start"], "hall": cfg["start2"]}
  life, body, _ = _flight()
  data, wall = life.data, time.time()
  out = {"seed": seed, "start": start, "exploreS": explore_s, "trips": trips, "restS": rest_s,
         "idleAt": idle_at, "idleS": idle_s, "learned": offset_learned}
  try:
    life.blacklist, life.floor_explored = set(), False
    body.start_at(*starts[start])
    out.update(draws(body, seed))
    if offset_learned:
      learned(body)
    tr = Tracker(body)
    out["walks"] = walks = []
    try:
      body.start_discovery()
      tr.phase = "look"
      body.run(body.look_around_routine())
      tr.phase = "explore"
      out["explored"] = life.explore(budget=explore_s, mark_done=False) if explore_s else None
      out["afterExplore"] = tr.off()
      sx, sy, _ = body.charge_standoff()
      if idle_at is not None:
        tr.phase = "to idle"
        t0 = float(data.time)
        there = bool(body.run(body.go_to_routine(*idle_at, timeout=WALK_TIMEOUT_S)))
        walks.append({"walk": tr.phase, "there": there, "s": round(float(data.time) - t0, 1),
                      "why": "" if there else life.drive_why(*idle_at), "off": tr.off(),
                      "zone": zone_of(*body.true_pose()[:2])})
        tr.phase = "idle"
        body.run(body.hold_routine(idle_s))
        walks.append({"walk": tr.phase, "off": tr.off()})
      for k in range(1, trips + 1):
        for leg, goal in (("out", STORE_XY), ("back", (sx, sy))):
          tr.phase = f"{leg}{k}"
          t0 = float(data.time)
          there = bool(body.run(body.go_to_routine(*goal, timeout=WALK_TIMEOUT_S)))
          walks.append({"walk": tr.phase, "there": there, "s": round(float(data.time) - t0, 1),
                        "why": "" if there else life.drive_why(*goal), "off": tr.off(),
                        "zone": zone_of(*body.true_pose()[:2])})
          if leg == "out" and rest_s:
            tr.phase = f"rest{k}"
            body.run(body.hold_routine(rest_s))
            walks.append({"walk": tr.phase, "off": tr.off()})
      if charge:
        tr.phase = "charge"
        t0 = float(data.time)
        out["docked"] = bool(life.go_charge())
        out.update(why=life.charge_failure, chargeS=round(float(data.time) - t0, 1),
                   atDock=tr.off())
        if out["docked"]:
          body.anchor_at_dock()           # as the loop does once it charges
          out["anchored"] = tr.off()
      # ...the last row at a tick of the reckoning, as every row is: between
      # two, the filter has moved on and the shares have not
      tr.due = 0.0
      body.run(body.hold_routine(0.05))
    except Stuck as stuck:
      out["stuck"] = {**stuck.args[0], "phase": tr.phase}
    out.update(rows=tr.rows, zones=tr.zone_table(), residual=tr.residual,
               counts=body.match_counts, falls=body.falls,
               events=[{k: e[k] for k in ("t", "why", "before", "after", "truth") if k in e}
                       for e in body.belief_events],
               simS=round(float(data.time), 1), wallS=round(time.time() - wall, 1))
  finally:
    body.close()
  return out


# ---- a fall (#572's stage 3) -----------------------------------------------------

#: `--falls`: shoved over after the explore -- this hard, N, this long, s
#: (`station_spike.shove`'s), from each of these directions off its heading,
#: deg -- its belief read as its get-up stands it, after it has stood this
#: long, s, and after a look round.
SHOVE_N, SHOVE_S = 300.0, 0.3
SHOVES_DEG = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
STOOD_S = 5.0


def fall(seed: str = "luca", start: str = "start", explore_s: float = 120.0,
         shove_deg: float = 0.0) -> dict:
  """One humanoid from `start`, its draws `seed`'s, the loop's explore for
  `explore_s`, then shoved over from `shove_deg` off its heading: how far
  its get-up truly moved it -- from the step it took the legs to the step
  it handed them back -- and its belief between those steps; its belief's
  error (m, deg) at each, `STOOD_S` after and after a look round; and the
  matcher's verdicts and the belief's records since the shove. A fall's
  get-up holds its odometry (#572): stood, the belief is off by what the
  get-up moved it, and the matcher's searches take that back. `stuck`, a
  fall it never got up from (`stuck_guard`), ends it there."""
  from pluggybot.lifecycle import HUMANOID_HOME, world_config
  cfg = world_config(HUMANOID_HOME)
  starts = {"start": cfg["start"], "hall": cfg["start2"]}
  life, body, _ = _flight()
  data, model, wall = life.data, life.model, time.time()
  m = body.mission

  def off():
    (bx, by, bh), (tx, ty, th) = body.pose, body.true_pose()
    return [round(math.hypot(bx - tx, by - ty), 3), round(math.degrees(wrap(bh - th)), 1)]

  out = {"seed": seed, "start": start, "exploreS": explore_s, "shoveDeg": shove_deg}
  try:
    life.blacklist, life.floor_explored = set(), False
    body.start_at(*starts[start])
    out.update(draws(body, seed))
    try:
      body.start_discovery()
      body.run(body.look_around_routine())
      life.explore(budget=explore_s, mark_done=False)
      out.update(before=off(), zone=zone_of(*body.true_pose()[:2]))
      x0, y0, yaw = body.true_pose()
      # ...the steps the get-up takes the legs and hands them back, read on
      # the physics seam, before any scan is matched after it
      seam = {}

      def at_seam():
        legs = m.legs
        if legs == "getup" and "fell" not in seam:
          seam["fell"] = (body.true_pose(), off(), body.pose)
        elif legs != "getup" and "fell" in seam and "stood" not in seam:
          seam["stood"] = (body.true_pose(), off(), body.pose)
      body.step_hooks.append(at_seam)
      counts0 = dict(body.match_counts)
      seq0 = max((e["seq"] for e in body.belief_events), default=0)
      push = np.array([math.cos(yaw + math.radians(shove_deg)),
                       math.sin(yaw + math.radians(shove_deg)), 0.0]) * SHOVE_N
      for _ in range(int(round(SHOVE_S / model.opt.timestep))):
        data.xfrc_applied[body.root, :3] = push
        body.run(body.hold_routine(model.opt.timestep))
      data.xfrc_applied[body.root] = 0.0
      t0 = float(data.time)
      while float(data.time) - t0 < GETUP_WAIT_S and not (m.falls and m.legs == "stand"):
        body.run(body.hold_routine(0.05))
      x1, y1, _ = body.true_pose()
      out.update(falls=m.falls, stood=m.falls > 0 and m.legs == "stand",
                 getupS=round(float(data.time) - t0, 1),
                 moved=round(math.hypot(x1 - x0, y1 - y0), 3))
      if "stood" in seam:
        (fx, fy, _), at_fall, (bx0, by0, _) = seam["fell"]
        (sx, sy, _), at_stood, (bx1, by1, _) = seam["stood"]
        out.update(getupMoved=round(math.hypot(sx - fx, sy - fy), 3),
                   beliefMoved=round(math.hypot(bx1 - bx0, by1 - by0), 3), atFall=at_fall,
                   atStood=at_stood)
      body.run(body.hold_routine(STOOD_S))
      out["afterStood"] = off()
      body.run(body.look_around_routine())
      out["afterLook"] = off()
      out["verdicts"] = {k: v - counts0.get(k, 0) for k, v in body.match_counts.items()
                         if v != counts0.get(k, 0)}
      out["events"] = [e["why"] for e in body.belief_events if e["seq"] > seq0]
    except Stuck as stuck:
      out.update(stuck=stuck.args[0], falls=m.falls, stood=False)
    out["wallS"] = round(time.time() - wall, 1)
  finally:
    body.close()
  return out


def falls(seed: str, start: str, explore_s: float, parallel: int) -> list[dict]:
  """`fall` from each of `SHOVES_DEG`, one process a fall, and the table."""
  import subprocess
  from concurrent.futures import ThreadPoolExecutor

  def run(deg):
    cmd = [sys.executable, __file__, "--one-fall",
           json.dumps({"seed": seed, "start": start, "explore": explore_s, "deg": deg})]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    try:
      return json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
      return {"seed": seed, "shoveDeg": deg, "crashed": r.stderr[-400:]}

  with ThreadPoolExecutor(parallel) as pool:
    rows = list(pool.map(run, SHOVES_DEG))
  print(f"{'shove':>5s} {'zone':12s} {'fell':>4s} {'up s':>5s} {'moved':>6s} {'get-up':>6s} "
        f"{'belief':>6s} {'off as it fell':>14s} {'as it stood':>13s} {f'{STOOD_S:g} s on':>13s} "
        f"{'after a look':>13s}  records")
  for r in rows:
    if "crashed" in r:
      print(f"{r['shoveDeg']:5.0f} crashed: {r['crashed'][-200:]}")
      continue
    if "stuck" in r:
      s = r["stuck"]
      print(f"{r['shoveDeg']:5.0f} {s['zone']:12s} {r['falls']:4d} stuck: fell at "
            f"t={s['fellT']:.0f} and not stood {s['t'] - s['fellT']:.0f} s on")
      continue
    print(f"{r['shoveDeg']:5.0f} {r['zone']:12s} {r['falls']:4d} {r['getupS']:5.1f} "
          f"{r['moved']:6.2f} {r.get('getupMoved', float('nan')):6.2f} "
          f"{r.get('beliefMoved', float('nan')):6.2f} "
          f"{str(r.get('atFall', '-')):>14s} {str(r.get('atStood', '-')):>13s} "
          f"{str(r['afterStood']):>13s} {str(r['afterLook']):>13s}  "
          f"{', '.join(r['events']) or '-'}")
  return rows


# ---- #593: standing where nothing is matched, and a run that ends first -------------

#: `--still`'s scenes before it stands: none since it booted, a walk this far
#: ahead, m, a turn on the spot this far, rad, and a stand-up's warp back to
#: its start after standing this long there, s (`HubLifecycle._stand_up`).
STILL_WALK_M, STILL_TURN_RAD, STILL_STANDUP_S = 2.0, 1.5, 60.0


def still(scene: str, seconds: float = 180.0, hold_after: float | None = None) -> dict:
  """One humanoid at the first robot's start, its scan matcher off and its
  gyro's offset the part's worst on every axis, standing still `seconds`
  after `scene` (`boot`, `walk`, `turn`, `standup`): what its body turned as
  it settled, and at each tick of its reckoning its believed heading --
  held at a standstill (`imu.HeadingHold`) from `hold_after` s of rest, the
  served `HOLD_AFTER_S` if None -- and its IMU's own, against the truth."""
  from pluggybot.humanoid import mission as hu
  from pluggybot.lifecycle import HUMANOID_HOME, world_config
  from pluggybot.perception.imu import GYRO_BIAS
  if hold_after is not None:
    hu.HOLD_AFTER_S = float(hold_after)
  life, body, _ = _life()
  data, wall = life.data, time.time()
  m = body.mission
  start = world_config(HUMANOID_HOME)["start"]

  def worst() -> None:                          # the part's worst, on a fresh IMU too
    m.stand.trunk.imu.gyro_bias[:] = (GYRO_BIAS, -GYRO_BIAS, GYRO_BIAS)

  try:
    body.start_at(*start)
    m.matcher = None
    worst()
    if scene == "walk":
      x, y, yaw = m.pose
      body.run(body.go_to_routine(x + STILL_WALK_M * math.cos(yaw),
                                  y + STILL_WALK_M * math.sin(yaw), timeout=30.0))
    elif scene == "turn":
      body.run(body.face_routine(m.pose[2] + STILL_TURN_RAD))
    elif scene == "standup":
      body.run(body.hold_routine(STILL_STANDUP_S))
      body.start_at(*start)
      worst()
    t0, (_, _, th0), b0 = float(data.time), m.true_pose(), m.odo.yaw
    a0, att = m.stand.trunk.attitude.yaw(), m.stand.trunk.attitude
    ticks: list[tuple] = []

    def tick() -> None:
      if (m._steps - 1) % m._odo_every:        # ...right after the reckoning's tick
        return
      turned = wrap(m.true_pose()[2] - th0)
      ticks.append((float(data.time) - t0, turned, wrap(m.odo.yaw - b0) - turned,
                    wrap(att.yaw() - a0) - turned, m._settled(), m.odo.hold.smooth))

    body.step_hooks.append(tick)
    body.run(body.hold_routine(seconds))
  finally:
    body.close()
  t, turned, held, unheld, settled, smooth = (np.array(v) for v in zip(*ticks))
  every = max(1, round(1.0 / max(t[1] - t[0], 1e-9)))
  # ...the rate the hold's gate reads while it may hold: its margin to `STILL_RATE`
  gate = np.degrees(np.abs(smooth[settled])) if settled.any() else np.zeros(1)
  return {"scene": scene, "seconds": seconds, "holdAfterS": hu.HOLD_AFTER_S, "legs": m.legs,
          "bodyTurnedDeg": round(math.degrees(turned[-1]), 2),
          "heldWorstDeg": round(math.degrees(np.abs(held).max()), 2),
          "unheldDeg": round(math.degrees(unheld[-1]), 2),
          "settledFromS": round(float(t[np.argmax(settled)]), 2) if settled.any() else None,
          "gateWorstDegS": round(float(gate.max()), 4),
          "gateP99DegS": round(float(np.percentile(gate, 99)), 4),
          "learnedDegS": round(math.degrees(m.odo.hold.bias), 4),
          "worstOffsetDegS": round(math.degrees(GYRO_BIAS), 4),
          "rows": [[round(float(t[k]), 2), round(math.degrees(turned[k]), 3),
                    round(math.degrees(held[k]), 3), round(math.degrees(unheld[k]), 3)]
                   for k in range(0, len(t), every)],
          "wallS": round(time.time() - wall, 1)}


def done(seconds: float = 300.0, wait: bool = True, hold: bool = True) -> dict:
  """The served pair in the house, the second robot's day over at once --
  its explore stubbed done, no work expected -- while the first's runs on:
  standing by for work, exploring, and charging past the run's `seconds`
  if it runs low. The second's belief against the truth every 10 s, and
  its matcher's verdicts, while it waits; without `wait` held STILL by the
  loop, without `hold` with no heading hold, as the loop was before #593."""
  from pluggybot import pair, tick
  from pluggybot.humanoid.mission import HumanoidMission
  from pluggybot.lifecycle import HUMANOID_HOME
  if not wait:
    run_many = tick.run_many
    tick.run_many = lambda pairs, name="", step=None, waits=None: run_many(pairs, name, step)
  if not hold:
    HumanoidMission._settled = lambda self: False
  wall = time.time()
  lives = pair.build_pair(HUMANOID_HOME, pack="demo", errands=("none", "none"))
  first, second = lives
  first.expects_work = True

  def explored(*args, **kw):
    second.floor_explored = True
    return "no-frontiers"
    yield

  second.explore_routine = explored
  rows: list[dict] = []
  due = [0.0]

  def sample() -> None:
    t = float(second.data.time)
    if t < due[0]:
      return
    due[0] = t + 10.0
    tx, ty, tyaw = second.body.true_pose()
    bx, by, byaw = second.body.pose
    rows.append({"t": round(t, 1), "state": second.state, "other": first.state,
                 "errM": round(math.hypot(bx - tx, by - ty), 3),
                 "headingDeg": round(math.degrees(wrap(byaw - tyaw)), 2),
                 "matched": dict(second.body.match_counts)})

  second.body.step_hooks.append(sample)
  ends = pair.run_pair(lives, max_sim_time=seconds, explore_budget=seconds)
  due[0] = 0.0
  sample()
  waiting = [r for r in rows if r["state"] == "DONE"]
  ok = [r["matched"].get("ok", 0) for r in waiting] or [0]
  return {"seconds": seconds, "wait": wait, "hold": hold,
          "doneFromS": waiting[0]["t"] if waiting else None, "endS": ends[0]["sim_time"],
          "headingWorstDeg": max((abs(r["headingDeg"]) for r in waiting), default=None),
          "errWorstM": max((r["errM"] for r in waiting), default=None),
          "matchedWaiting": ok[-1] - ok[0], "rows": rows, "wallS": round(time.time() - wall, 1)}


def still_report(run: dict) -> None:
  print(f"== --still {run['scene']}, the hold from {run['holdAfterS']:.0f} s of rest "
        f"(it began at {run['settledFromS']} s), the legs the {run['legs']}'s: its body turned "
        f"{run['bodyTurnedDeg']:+.2f} deg as it settled; held, off by at most "
        f"{run['heldWorstDeg']:.2f} deg; unheld, {run['unheldDeg']:+.2f} at {run['seconds']:.0f} s; "
        f"its gate read at most {run['gateWorstDegS']:.4f} deg/s (p99 {run['gateP99DegS']:.4f}); "
        f"it learned {run['learnedDegS']:.4f} deg/s (the part's worst {run['worstOffsetDegS']:.4f}); "
        f"{run['wallS']:.0f} s")
  print(f"  {'t':>6s} {'body turned':>11s} {'held off':>8s} {'unheld off':>10s} deg")
  for t, turned, held, unheld in run["rows"][::10]:
    print(f"  {t:6.1f} {turned:+11.3f} {held:+8.3f} {unheld:+10.3f}")


def done_report(run: dict) -> None:
  print(f"== --done{'' if run['wait'] else ' --no-wait'}{'' if run['hold'] else ' --no-hold'}: "
        f"the second robot's day over at {run['doneFromS']} s, the run's at {run['endS']:.3f} s; "
        f"waiting, its heading off by at most {run['headingWorstDeg']} deg and its belief "
        f"{run['errWorstM']} m, {run['matchedWaiting']} scans matched; {run['wallS']:.0f} s")
  for r in run["rows"][::5]:
    print(f"  t={r['t']:6.1f} {r['state']:9s} (the first {r['other']:9s}) "
          f"{r['errM']:6.3f} m {r['headingDeg']:+7.2f} deg  matched {r['matched']}")


# ---- the report -----------------------------------------------------------------


def _norm(v) -> float:
  return float(math.hypot(*v))


def report(run: dict) -> None:
  """One run: its draws, how each walk ended, the error and its shares at
  each phase's end, and what each zone added."""
  rate = run["gyroOffsetDegS"][2] * 60
  taken = run.get("learned")
  print(f"== seed {run['seed']} from {run['start']}: gyro offset {rate:+.2f} deg/min about "
        f"the vertical{' (taken off its heading: --learned)' if taken else ''}, scale "
        f"{run['gyroScalePct'][2]:+.3f} %; {run['simS']:.0f} sim s in "
        f"{run['wallS']:.0f} s; residual {run['residual'][0]:.1e} m, "
        f"{math.degrees(run['residual'][1]):.1e} deg; falls {run['falls']}")
  stuck = run.get("stuck")
  if stuck:
    print(f"  STUCK: fell at t={stuck['fellT']:.0f} in the {stuck['zone']} ({stuck['phase']}) "
          f"and not stood {stuck['t'] - stuck['fellT']:.0f} s on: the flight ended there")
  if not run["rows"]:
    return
  last = run["rows"][-1]
  offset = 0.0 if taken else run["gyroOffsetDegS"][2] * last["t"]
  scale = run["gyroScalePct"][2] / 100 * last["turnedDeg"]
  print(f"  the gyro's share {last['turn']['gyro']:+.1f} deg over {last['t']:.0f} s: its offset "
        f"alone {offset:+.1f}, its scale over the {last['turnedDeg']:+.0f} deg truly turned "
        f"{scale:+.1f}, the rest {last['turn']['gyro'] - offset - scale:+.1f}; the hold's "
        f"{last['turn'].get('hold', 0.0):+.1f}")
  if "explored" in run:
    print(f"  explore {run['exploreS']:.0f} s: {run['explored']}, ended {run['afterExplore']} off")
  for w in run["walks"]:
    print(f"  {w['walk']:7s} " + (f"{'there' if w['there'] else 'NOT THERE'} in {w['s']:.0f} s, "
                                  f"in the {w['zone']}" if "there" in w else "stood")
          + f", {w['off'][0]:.2f} m {w['off'][1]:+.1f} deg off"
          + (f" -- {w['why']}" if w.get("why") else ""))
  if "docked" in run:
    print(f"  charge: {'docked' if run['docked'] else 'NOT docked: ' + run['why']} in "
          f"{run['chargeS']:.0f} s, {run['atDock']} off" +
          (f", {run['anchored']} after the station's fix" if "anchored" in run else ""))
  print(f"  {'at the end of':12s} {'t':>6s} {'belief m/deg':>13s} {'odom alone':>13s} "
        f"{'legs alone':>10s} | {'legs':>5s} {'heading':>7s} {'matcher':>7s} {'station':>7s} m "
        f"| {'gyro':>6s} {'hold':>6s} {'matcher':>7s} {'station':>7s} deg")
  ends = [r for k, r in enumerate(run["rows"])
          if k + 1 == len(run["rows"]) or run["rows"][k + 1]["phase"] != r["phase"]]
  for r in ends:
    s, t = r["share"], r["turn"]
    held = t.get("hold", 0.0)                # ...none in a run from before #593
    print(f"  {r['phase']:12s} {r['t']:6.0f} {r['errM']:6.2f}/{r['errDeg']:+6.1f} "
          f"{r['odoM']:6.2f}/{t['gyro'] + held:+6.1f} {r['legsM']:10.2f} | "
          f"{_norm(s['legs']):5.2f} {_norm(s['heading']):7.2f} {_norm(s['matcher']):7.2f} "
          f"{_norm(s['station']):7.2f}   | {t['gyro']:+6.1f} {held:+6.1f} {t['matcher']:+7.1f} "
          f"{t['station']:+7.1f}")
  worst = max(run["rows"], key=lambda r: r["errM"])
  print(f"  worst {worst['errM']:.2f} m ({worst['errDeg']:+.1f} deg) at t={worst['t']:.0f} in "
        f"the {worst['zone']}, {worst['phase']}")


#: The zones grouped as the table reads them.
INSIDE = ("living", "bedroom", "hall", "kitchen", "workshop")
FACILITY = ("lobby", "lab", "store")


def zone_report(runs: list[dict]) -> None:
  """What each zone added over `runs`, per minute there and per metre
  walked there: the gyro's heading, the hold's and the matcher's, the legs'
  position, the believed heading's and the matcher's -- and the matcher's
  verdicts and its returns a scan."""
  tot: dict[str, dict] = {}
  for run in runs:
    for z in run["zones"]:
      t = tot.setdefault(z["zone"], {"s": 0.0, "walkedM": 0.0, "verdicts": {}, "scans": 0,
                                     "returns": 0.0, "gyroDeg": 0.0, "holdDeg": 0.0,
                                     "matcherDeg": 0.0, "stationDeg": 0.0, "legsM": 0.0,
                                     "headingM": 0.0, "matcherM": 0.0})
      t["s"] += z["s"]
      t["walkedM"] += z["walkedM"]
      t["scans"] += z["scans"]
      t["returns"] += z["returnsPerScan"] * z["scans"]
      for k in ("gyroDeg", "holdDeg", "matcherDeg", "stationDeg"):
        t[k] += abs(z.get(k, 0.0))           # ...no hold in a run from before #593
      for k in ("legsM", "headingM", "matcherM"):
        t[k] += _norm(z[k])
      for k, v in z["verdicts"].items():
        t["verdicts"][k] = t["verdicts"].get(k, 0) + v
  print(f"== by zone, {len(runs)} runs: each run's own sum in the zone, its size, added up "
        "(a share that grew and shrank there reads small)")
  print(f"  {'zone':16s} {'min':>5s} {'m':>6s} | {'gyro':>5s} {'hold':>5s} {'match':>5s} "
        f"deg/min | {'legs':>5s} {'head':>5s} {'match':>5s} cm/m | {'ok':>4s} {'held':>4s} "
        f"{'none':>4s} {'refused':>7s} % of scans | returns")
  order = sorted(tot, key=lambda n: (n not in INSIDE, n in FACILITY, n))
  for name in order:
    t = tot[name]
    mins, metres = t["s"] / 60, max(t["walkedM"], 1e-9)
    v, n = t["verdicts"], max(t["scans"], 1)
    none = sum(v.get(k, 0) for k in ("no map", "sparse", "unconstrained"))
    refused = sum(v.get(k, 0) for k in ("inconsistent", "slid"))
    print(f"  {name:16s} {mins:5.1f} {t['walkedM']:6.1f} | {t['gyroDeg'] / max(mins, 1e-9):5.2f} "
          f"{t['holdDeg'] / max(mins, 1e-9):5.2f} "
          f"{t['matcherDeg'] / max(mins, 1e-9):5.2f}         | "
          f"{100 * t['legsM'] / metres:5.1f} {100 * t['headingM'] / metres:5.1f} "
          f"{100 * t['matcherM'] / metres:5.1f}      | {100 * v.get('ok', 0) / n:4.0f} "
          f"{100 * v.get('held', 0) / n:4.0f} {100 * none / n:4.0f} {100 * refused / n:7.0f}"
          f"            | {t['returns'] / n:5.0f}")


def table(runs: list[dict]) -> None:
  """Runs side by side: the error after the explore, its worst and where,
  at the storeroom and back at the standoff each trip, and the dock."""
  print(f"{'seed':>6s} {'from':>5s} {'deg/min':>7s} {'explore':>13s} {'worst':>13s} "
        f"{'where worst':14s} {'out / back, m and deg':44s} docked")
  for run in runs:
    worst = max(run["rows"], key=lambda r: r["errM"])
    walks = " ".join(f"{w['off'][0]:.1f}/{w['off'][1]:+.0f}" for w in run["walks"]
                     if "there" in w)
    docked = "stuck" if run.get("stuck") else run.get("docked")
    explored = (f"{run['afterExplore'][0]:6.2f}/{run['afterExplore'][1]:+6.1f}"
                if "afterExplore" in run else f"{'-':>13s}")
    seed = run["seed"] + (" L" if run.get("learned") else "")
    print(f"{seed:>6s} {run['start']:>5s} {60 * run['gyroOffsetDegS'][2]:+7.2f} {explored} "
          f"{worst['errM']:6.2f}/{worst['errDeg']:+6.1f} {worst['zone']:14s} {walks:44s} "
          f"{'-' if docked is None else docked}")
  if any(run.get("learned") for run in runs):
    print("  L: its gyro's offset taken off its heading (--learned)")


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--seed", default="luca", help="the IMU's and joints' draws: luca, rowan, or any")
  ap.add_argument("--start", default="start", choices=("start", "hall"))
  ap.add_argument("--explore", type=float, default=240.0, help="the explore's budget, s")
  ap.add_argument("--trips", type=int, default=2, help="walks to the storeroom and back")
  ap.add_argument("--rest", type=float, default=0.0, help="standing still at the storeroom, s")
  ap.add_argument("--idle-at", default=None, metavar="X,Y",
                  help="after the explore, walk here and stand --idle s first")
  ap.add_argument("--idle", type=float, default=180.0, help="standing idle at --idle-at, s")
  ap.add_argument("--learned", action="store_true",
                  help="the gyro's offset taken off its heading, as a zero-rate update "
                       "that had learned it would")
  ap.add_argument("--no-charge", action="store_true", help="stop after the last walk")
  ap.add_argument("--falls", action="store_true",
                  help="after an explore of --fall-explore s, shoved over six ways, a "
                       "process each: the belief as it stands and after (#572)")
  ap.add_argument("--fall-explore", type=float, default=120.0)
  ap.add_argument("--parallel", type=int, default=3, help="--falls' processes at once")
  ap.add_argument("--out", default=None, help="the run, as JSON")
  ap.add_argument("--table", nargs="+", default=None, metavar="JSON",
                  help="runs written by --out, side by side and by zone")
  ap.add_argument("--still", default=None, choices=("boot", "walk", "turn", "standup"),
                  help="#593: standing still after it, its matcher off and its gyro's "
                       "offset the part's worst: its heading held and unheld")
  ap.add_argument("--hold-after", type=float, default=None, metavar="S",
                  help="--still: the hold may begin this long into a rest (served: "
                       "`HOLD_AFTER_S`)")
  ap.add_argument("--done", action="store_true",
                  help="#593: the pair, the second robot's day over first; its belief "
                       "while it waits")
  ap.add_argument("--no-wait", action="store_true",
                  help="--done: a finished robot held STILL by the loop, as before #593")
  ap.add_argument("--no-hold", action="store_true",
                  help="--done: no heading hold at a standstill, as before #593")
  ap.add_argument("--seconds", type=float, default=None,
                  help="--still: how long it stands (180); --done: the run's budget (300)")
  ap.add_argument("--one-fall", default=None, help=argparse.SUPPRESS)
  args = ap.parse_args(argv)
  if args.one_fall:
    t = json.loads(args.one_fall)
    print(json.dumps(fall(t["seed"], t["start"], t["explore"], t["deg"])))
    return 0
  if args.falls:
    rows = falls(args.seed, args.start, args.fall_explore, args.parallel)
    if args.out:
      Path(args.out).write_text(json.dumps(rows))
    return 0
  if args.still or args.done:
    if args.still:
      run = still(args.still, args.seconds or 180.0, args.hold_after)
      still_report(run)
    else:
      run = done(args.seconds or 300.0, wait=not args.no_wait, hold=not args.no_hold)
      done_report(run)
    if args.out:
      Path(args.out).write_text(json.dumps(run))
    return 0
  if args.table:
    runs = [json.loads(Path(p).read_text()) for p in args.table]
    for run in runs:
      report(run)
    table(runs)
    zone_report(runs)
    return 0
  idle_at = None if args.idle_at is None else tuple(float(v) for v in args.idle_at.split(","))
  run = fly(args.seed, args.start, args.explore, args.trips, args.rest, not args.no_charge,
            idle_at, args.idle if idle_at is not None else 0.0, args.learned)
  if args.out:
    Path(args.out).write_text(json.dumps(run))
  report(run)
  zone_report([run])
  return 0


if __name__ == "__main__":
  sys.exit(main())
