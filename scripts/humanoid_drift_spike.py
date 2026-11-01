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
            is its IMU's alone (`humanoid.odometry`), so its legs put none
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

`--seed` draws the IMU's offset and scale and the joints' noise: `luca` and
`rowan` are the served pair's own (the first robot's and the second's),
anything else a draw keyed by it. The truth is ours, read to take the
error apart; nothing here steers off it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from station_spike import _life, zone_of  # noqa: E402  (the house as the station's trials fly it)

#: A row of the belief against the truth this often, sim s.
EVERY_S = 5.0
#: The walk across the street goes to the storeroom's middle, in front of
#: its counter (`home.storeroom`), and back to the station's standoff, each
#: walk given this long, s.
STORE_XY = (25.0, -3.0)
WALK_TIMEOUT_S = 300.0
#: The position's shares and the heading's, in the order they are said;
#: `set` is any other move of the belief (none in a flight here).
SHARES = ("legs", "heading", "matcher", "station", "set")
TURNS = ("gyro", "matcher", "station", "set")


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
  m.stand.joints.rng = np.random.default_rng(zlib.crc32(f"joints:{key}".encode()))
  return {"gyroOffsetDegS": [round(math.degrees(v), 4) for v in imu.gyro_bias],
          "gyroScalePct": [round(100 * (v - 1), 3) for v in imu.gyro_scale]}


def learned(body) -> None:
  """The gyro's offset taken off what its heading filter integrates, as a
  zero-rate update that had learned it would (`imu.Standstill`, the
  quadruped's: nothing under `humanoid/` has one, #572) -- learned
  exactly, at once. The walking policy still reads the gyro raw, as the
  quadruped's did: only the reckoning was changed."""
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
    through the heading it believed, against the truth's."""
    bx, by, byaw = before
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
    g = a + self.gyro_ref
    self.odo_only += (math.cos(g) * ahead - math.sin(g) * left,
                      math.sin(g) * ahead + math.cos(g) * left)
    self.legs_only += laid
    self.shares["legs"] += legs
    self.shares["heading"] += heading
    self.turns["gyro"] += gyro
    walked = math.hypot(tx - px, ty - py)
    self.walked, self.turned = self.walked + walked, self.turned + turned
    t = float(self.d.time)
    z = self._zone()
    z["s"] += t - self.t
    z["walkedM"] += walked
    z["turnedRad"] += turned
    z["gyro"] += gyro
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
    before = (odo.x, odo.y, odo.yaw)
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
  shares over the flight and over each zone."""
  from pluggybot.lifecycle import HUMANOID_HOME, world_config
  cfg = world_config(HUMANOID_HOME)
  starts = {"start": cfg["start"], "hall": cfg["start2"]}
  life, body, _ = _life()
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
    body.start_discovery()
    tr.phase = "look"
    body.run(body.look_around_routine())
    tr.phase = "explore"
    out["explored"] = life.explore(budget=explore_s, mark_done=False) if explore_s else None
    out["afterExplore"] = tr.off()
    sx, sy, _ = body.charge_standoff()
    walks = []
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
    out["walks"] = walks
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
#: ...waiting for its get-up this long at most, s (`HumanoidBody.GETUP_BUDGET_S`'s
#: margin: the slowest in the house stood at 30).
GETUP_WAIT_S = 40.0


def fall(seed: str = "luca", start: str = "start", explore_s: float = 120.0,
         shove_deg: float = 0.0) -> dict:
  """One humanoid from `start`, its draws `seed`'s, the loop's explore for
  `explore_s`, then shoved over from `shove_deg` off its heading: how far
  its get-up truly moved it -- from the step it took the legs to the step
  it handed them back -- and its belief between those steps; its belief's
  error (m, deg) at each, `STOOD_S` after and after a look round; and the
  matcher's verdicts and the belief's records since the shove. A fall's
  get-up holds its odometry (#572): stood, the belief is off by what the
  get-up moved it, and the matcher's searches take that back."""
  from pluggybot.lifecycle import HUMANOID_HOME, world_config
  cfg = world_config(HUMANOID_HOME)
  starts = {"start": cfg["start"], "hall": cfg["start2"]}
  life, body, _ = _life()
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
    print(f"{r['shoveDeg']:5.0f} {r['zone']:12s} {r['falls']:4d} {r['getupS']:5.1f} "
          f"{r['moved']:6.2f} {r.get('getupMoved', float('nan')):6.2f} "
          f"{r.get('beliefMoved', float('nan')):6.2f} "
          f"{str(r.get('atFall', '-')):>14s} {str(r.get('atStood', '-')):>13s} "
          f"{str(r['afterStood']):>13s} {str(r['afterLook']):>13s}  "
          f"{', '.join(r['events']) or '-'}")
  return rows


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
  last = run["rows"][-1]
  offset = 0.0 if taken else run["gyroOffsetDegS"][2] * last["t"]
  scale = run["gyroScalePct"][2] / 100 * last["turnedDeg"]
  print(f"  the gyro's share {last['turn']['gyro']:+.1f} deg over {last['t']:.0f} s: its offset "
        f"alone {offset:+.1f}, its scale over the {last['turnedDeg']:+.0f} deg truly turned "
        f"{scale:+.1f}, the rest {last['turn']['gyro'] - offset - scale:+.1f}")
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
        f"| {'gyro':>6s} {'matcher':>7s} {'station':>7s} deg")
  ends = [r for k, r in enumerate(run["rows"])
          if k + 1 == len(run["rows"]) or run["rows"][k + 1]["phase"] != r["phase"]]
  for r in ends:
    s, t = r["share"], r["turn"]
    print(f"  {r['phase']:12s} {r['t']:6.0f} {r['errM']:6.2f}/{r['errDeg']:+6.1f} "
          f"{r['odoM']:6.2f}/{t['gyro']:+6.1f} {r['legsM']:10.2f} | {_norm(s['legs']):5.2f} "
          f"{_norm(s['heading']):7.2f} {_norm(s['matcher']):7.2f} {_norm(s['station']):7.2f}   "
          f"| {t['gyro']:+6.1f} {t['matcher']:+7.1f} {t['station']:+7.1f}")
  worst = max(run["rows"], key=lambda r: r["errM"])
  print(f"  worst {worst['errM']:.2f} m ({worst['errDeg']:+.1f} deg) at t={worst['t']:.0f} in "
        f"the {worst['zone']}, {worst['phase']}")


#: The zones grouped as the table reads them.
INSIDE = ("living", "bedroom", "hall", "kitchen", "workshop")
FACILITY = ("lobby", "lab", "store")


def zone_report(runs: list[dict]) -> None:
  """What each zone added over `runs`, per minute there and per metre
  walked there: the gyro's heading and the matcher's, the legs' position,
  the believed heading's and the matcher's -- and the matcher's verdicts
  and its returns a scan."""
  tot: dict[str, dict] = {}
  for run in runs:
    for z in run["zones"]:
      t = tot.setdefault(z["zone"], {"s": 0.0, "walkedM": 0.0, "verdicts": {}, "scans": 0,
                                     "returns": 0.0, "gyroDeg": 0.0, "matcherDeg": 0.0,
                                     "stationDeg": 0.0, "legsM": 0.0, "headingM": 0.0,
                                     "matcherM": 0.0})
      t["s"] += z["s"]
      t["walkedM"] += z["walkedM"]
      t["scans"] += z["scans"]
      t["returns"] += z["returnsPerScan"] * z["scans"]
      for k in ("gyroDeg", "matcherDeg", "stationDeg"):
        t[k] += abs(z[k])
      for k in ("legsM", "headingM", "matcherM"):
        t[k] += _norm(z[k])
      for k, v in z["verdicts"].items():
        t["verdicts"][k] = t["verdicts"].get(k, 0) + v
  print(f"== by zone, {len(runs)} runs: each run's own sum in the zone, its size, added up "
        "(a share that grew and shrank there reads small)")
  print(f"  {'zone':16s} {'min':>5s} {'m':>6s} | {'gyro':>5s} {'match':>5s} deg/min | "
        f"{'legs':>5s} {'head':>5s} {'match':>5s} cm/m | {'ok':>4s} {'held':>4s} {'none':>4s} "
        f"{'refused':>7s} % of scans | returns")
  order = sorted(tot, key=lambda n: (n not in INSIDE, n in FACILITY, n))
  for name in order:
    t = tot[name]
    mins, metres = t["s"] / 60, max(t["walkedM"], 1e-9)
    v, n = t["verdicts"], max(t["scans"], 1)
    none = sum(v.get(k, 0) for k in ("no map", "sparse", "unconstrained"))
    refused = sum(v.get(k, 0) for k in ("inconsistent", "slid"))
    print(f"  {name:16s} {mins:5.1f} {t['walkedM']:6.1f} | {t['gyroDeg'] / max(mins, 1e-9):5.2f} "
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
    docked = run.get("docked")
    seed = run["seed"] + (" L" if run.get("learned") else "")
    print(f"{seed:>6s} {run['start']:>5s} {60 * run['gyroOffsetDegS'][2]:+7.2f} "
          f"{run['afterExplore'][0]:6.2f}/{run['afterExplore'][1]:+6.1f} "
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
