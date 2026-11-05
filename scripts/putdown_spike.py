"""The table's put-downs, measured before the waist is built (#560, stage (c2)).

#620's instrument (`scripts/waist_spike.py`) found the place's bottleneck:
our procedure's carry into a put-down near the body's middle -- the plate's
wall set down 0.30 m ahead and 8 cm left of the middle, the fork 16 cm --
stays out of reach with any waist (SimNotes, "A waist, measured before it is
built"). This measures the LEVERS that might open it, each on the robot's
model of itself as that instrument measures a reach (`ws.plan`, a statue's
legs, the arm's own solve), on today's body (`none`), the waist yaw Ben
chose (`yaw`, +-60 deg) and the same yaw at +-90 (`yaw90`):

  the grip     tilted 30, 45 (our procedure's), 60 or 90 deg (fingers
               down), its jaws either way round; the plate's wall gripped
               at another point; the grip turning on its way
  the path     lifted higher; out and round, or the put-down approached
               from ahead, through a `via` point
  the hand     the other hand; a relay: the piece set down halfway and
               picked up again from another stance, by either hand (a
               regrasp, where it is the other)
  the stance   nearer the table or further from it; turned toward the piece
  the layout   where the dishware waits beside the place

For each, the WINDOW: the stances, 1 cm apart across, from which the hand
picks the piece up (`ws._ok`: reached, clear of the table and of its own
parts, still reached once balanced again) and carries it where it goes and
sets it down, clear, by three measures of the carry:

  one        one posture follows it all at the reach's own tolerance, the
             reach's postures: #620's calibrated window (`ws.carry_plan`)
  repose     the posture changed on the way with the hand held still, the
             reach's postures: #620's `--reposture` search
             (`ws.reposture_carry`), its points clear
  served     our procedure's verbs as the served reach makes them, every
             posture one of the carry's own (`reach.HOLD_POSTURES`, three
             bows at full height more than the reach's): each `pull` in
             the posture `path_for` plans, a straight `move_hand` in the one
             `posture_for` finds for its end and taking another where it
             misses on its way, a `via` path or a turn in the one
             `path_for` plans for it all (`served`)

THE BAR, written before anything was measured (#560's brief): every piece
picked up and carried from a window at least 6 cm wide -- twice a
sidestep's +-3 cm -- from stances our procedure reaches by steps. Every
number is a STATUE's (#620): compare statue to statue, and confirm a lever
by flying it.

  --levers A,B     which levers (default all, `LEVERS`); --list prints them
  --candidates     which bodies (default none,yaw,yaw90)
  --pieces         which pieces (default all)
  --into DIR       DIR/rows.jsonl, a row a stance; --read DIR the report
  --jobs N         processes (default 3): a pod's batch for the whole set
  --sample N       every Nth task (a smoke); --part I/N the Ith of N shares
                   (a pod each: `--read A,B` reads their records together);
                   --models M,N these measures alone (a re-run of one)
  --check          the base lever against #620's window rows, cell by cell
"""

import argparse
import math
import multiprocessing as mp
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import waist_spike as ws  # noqa: E402  (#620's instrument: the candidates, the planner, the checks)

from pluggybot.challenge import table as tb  # noqa: E402
from pluggybot.evaluation import counter as ct  # noqa: E402
from pluggybot.humanoid import arms as ha  # noqa: E402
from pluggybot.humanoid import model as hm  # noqa: E402
from pluggybot.humanoid import parts as hp  # noqa: E402
from pluggybot.humanoid import reach as rc  # noqa: E402

KINDS = ("none", "yaw", "yaw90")

# ---- the place, in the start's stance frame ----------------------------------
#
# The robot's soles' middle `ct.TABLE_GAP` out of the table's edge on the
# place's line, facing in: x ahead, y to its left (the place's T frame moved
# back by the gap). Each piece WAITS at its middle (`ct.TABLE_SOURCES`) and
# GOES to its spot (`tb.SPOTS`, from the card's foot toward the diner).

def _start(t) -> tuple[float, float]:
  return (round(ct.TABLE_GAP + t[0], 4), round(t[1], 4))


def _spot(name: str) -> tuple[float, float]:
  a, b = tb.SPOTS[name]
  return (round(ct.TABLE_GAP + ct.CARD_IN - a, 4), round(b, 4))


SOURCES = {n: _start(ct.TABLE_SOURCES[n][0]) for n in ct.TABLE_SOURCES}
SPOT = {n: _spot(n) for n in tb.SPOTS}
#: Where our procedure (`hands_check.TABLE`) grips each, m: the plate this far
#: from its middle, at its wall; and the grip's height over the floor (the
#: plate's wall and the fork's handle at 0.767 and 0.762, the cup at its
#: lid's tag less 3 cm).
PLATE_GRIP_M = 0.081
GRIP_Z = {n: ws.PIECES[n][1][2] for n in ws.PIECES}
#: The hand our procedure takes each with.
HAND = {n: ws.PIECES[n][0] for n in ws.PIECES}

#: The dishware's other layouts (`ct.TABLE_SOURCES`' T frame, each piece's
#: middle): Ben's call 4 lets the restaurant change when the job is offered.
#: `across` waits level with its spot, beside it, and is carried straight
#: across; `near` waits nearer the table's edge, and is carried outward and
#: across -- the carry running away from the body, never in toward it.
LAYOUTS = {
  "today": {n: ct.TABLE_SOURCES[n][0] for n in ct.TABLE_SOURCES},
  "across": {"plate": (0.17, 0.27), "fork": (0.17, 0.42), "cup": (0.29, -0.31)},
  "near": {"plate": (0.10, 0.24), "fork": (0.06, 0.38), "cup": (0.29, -0.31)},
}


def grip_turn(tilt_deg: float, jaws: tuple[float, float], lean: tuple[float, float]) -> np.ndarray:
  """A grip, stance frame: its jaws closing along `jaws` (level), its
  fingers leaning along `lean` (level, across the jaws) and `tilt_deg`
  below level -- 90 straight down."""
  c, s = math.cos(math.radians(tilt_deg)), math.sin(math.radians(tilt_deg))
  return ha.frame((c * lean[0], c * lean[1], -s), (jaws[0], jaws[1], 0.0))


def _turned(v, deg: float):
  c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
  return (c * v[0] - s * v[1], s * v[0] + c * v[1])


@dataclass(frozen=True)
class Lever:
  """How a piece is picked up and carried (module docstring). Every field
  at its default is our procedure, as `ws.PIECES` and `--window` measure it."""
  name: str
  note: str = ""
  #: the grip's tilt below level at the pick-up and, where it turns on its
  #: way (`move_hand`'s turn), at the put-down, deg
  tilt: float = 45.0
  put_tilt: float | None = None
  #: the jaws as asked: +1 our procedure's (closing to the left), -1 the
  #: other way round; and whether the reach may take the other where the
  #: asked plans nothing (`Reach._planned`), as our procedure's does
  jaws: int = 1
  flip: bool = True
  #: where the plate's wall is gripped, deg round its middle from straight
  #: ahead (90: its left side, our procedure's) -- and where it is at the
  #: put-down, the hand turning about the upright on its way (`move_hand`'s
  #: turn: the plate is round, and turns with it); and how far along the
  #: fork's handle toward its head, m
  rim: float = 90.0
  put_rim: float | None = None
  along: float = 0.0
  #: the lift before the carry, m, and the carry's way between: straight,
  #: `round` (out ahead of the middle of the way by `out`), or `ahead` (to
  #: `out` ahead of the put-down, then in to it)
  lift: float = ws.LIFT_M
  path: str = "straight"
  out: float = 0.08
  #: the hand (None: our procedure's for the piece), and the stance's place:
  #: `dx` nearer the table, `turn` deg to the left (`step`'s turn, then its
  #: walk across)
  hand: str | None = None
  dx: float = 0.0
  turn: float = 0.0
  layout: str = "today"
  #: a RELAY: the piece set down here first (its middle, start frame) by
  #: one stance and picked up again by another, its hand `relay_hand` (None:
  #: the same), carried the rest
  relay: tuple | None = None
  relay_hand: str | None = None
  pieces: tuple = ("plate", "fork", "cup")
  #: which carries are measured beside `one`
  models: tuple = ("one", "repose", "served")


def _levers() -> dict:
  out = [Lever("base", "our procedure: 45 deg, the jaws as it asks, a 5 cm lift, straight")]
  out.append(Lever("jaws-", "the jaws the other way round", jaws=-1))
  for j, w in ((1, "as asked"), (-1, "the other way round")):
    out.append(Lever(f"only{'+' if j > 0 else '-'}", f"the jaws {w} and never flipped", jaws=j,
                     flip=False))
  for t in (30.0, 60.0, 90.0):
    for j in (1, -1):
      out.append(Lever(f"t{t:.0f}{'+' if j > 0 else '-'}", f"tilted {t:.0f} deg, the jaws "
                       f"{'as asked' if j > 0 else 'the other way round'}", tilt=t, jaws=j,
                       pieces=("plate", "fork")))
  for t in (30.0, 60.0, 90.0):
    for j in (1, -1):
      out.append(Lever(f"to{t:.0f}{'+' if j > 0 else '-'}",
                       f"picked up at 45 deg, turned to {t:.0f} on its way (`move_hand`'s turn)",
                       put_tilt=t, jaws=j, flip=False, pieces=("plate", "fork"),
                       models=("one", "served")))
  for rim in (60.0, 120.0, 45.0, 135.0, 30.0, 150.0):
    for t in (45.0, 90.0):
      out.append(Lever(f"rim{rim:.0f}t{t:.0f}", f"the plate's wall gripped {rim:.0f} deg round "
                       f"from straight ahead (90 its left side), tilted {t:.0f}", rim=rim,
                       tilt=t, pieces=("plate",)))
  for rim in (45.0, 60.0, 120.0, 135.0):
    for j in (1, -1):
      # ⚠ our procedure's plate grip plans only the other way round (`only+`
      # carries nothing), so the jaws as asked never flipped measure nothing
      out.append(Lever(f"spin{rim:.0f}{'' if j > 0 else '-'}", f"the plate picked up by its left "
                       f"side and turned on its way about the upright, set down gripped {rim:.0f} "
                       f"deg round, the jaws {'as asked' if j > 0 else 'the other way round'}",
                       put_rim=rim, jaws=j, flip=False, pieces=("plate",),
                       models=("one", "served")))
  for a in (-0.03, 0.03):
    out.append(Lever(f"along{a * 100:+.0f}", f"the fork's handle gripped {a * 100:+.0f} cm "
                     "toward its head", along=a, pieces=("fork",)))
  for lift in (0.10, 0.15):
    out.append(Lever(f"lift{lift * 100:.0f}", f"lifted {lift * 100:.0f} cm", lift=lift,
                     pieces=("plate", "fork")))
  for path in ("round", "ahead"):
    for o in (0.05, 0.10):
      out.append(Lever(f"{path}{o * 100:.0f}", ("out and round, " if path == "round" else
                       "the put-down approached from ahead, ") + f"{o * 100:.0f} cm out",
                       path=path, out=o, pieces=("plate", "fork"),
                       models=("one", "served")))
  out.append(Lever("right", "the other hand: the plate's right wall, the fork by the right",
                   hand="right", pieces=("plate", "fork")))
  out.append(Lever("right-", "the other hand, its jaws the other way round", hand="right",
                   jaws=-1, pieces=("plate", "fork")))
  for dx in (-0.10, -0.05, 0.05, 0.10):
    out.append(Lever(f"dx{dx * 100:+.0f}", f"the stance {abs(dx) * 100:.0f} cm "
                     f"{'nearer the table (under it, knee room)' if dx > 0 else 'further back'}",
                     dx=dx))
  for turn in (-30.0, -20.0, -10.0, 10.0, 20.0, 30.0):
    out.append(Lever(f"turn{turn:+.0f}deg", f"the stance turned {abs(turn):.0f} deg to the "
                     f"{'left' if turn > 0 else 'right'} (`step`'s turn)", turn=turn,
                     pieces=("plate", "fork")))
  for lay in ("across", "near"):
    out.append(Lever(f"layout-{lay}", f"the dishware waiting `{lay}` (`LAYOUTS`)", layout=lay))
  # a relay: the piece set down partway, then picked up again from another
  # stance -- by the same hand, or by the other (a regrasp)
  for piece in ("plate", "fork"):
    at = tuple(round((a + b) / 2, 4) for a, b in zip(SOURCES[piece], SPOT[piece]))
    for name, hand in ((f"relay-{piece}", None), (f"relay-{piece}-right", "right")):
      out.append(Lever(name, f"the {piece} set down halfway, {at}, then carried on "
                       f"{'by the right hand' if hand else 'from another stance'}", relay=at,
                       relay_hand=hand, pieces=(piece,)))
  return {lev.name: lev for lev in out}


LEVERS = _levers()

# ---- a lever's carries, in a stance's frame ----------------------------------


@dataclass
class Carry:
  """One carry of a piece, start frame: picked up at `pick` with the grip
  `rot` by `side`, set down at `put` with the grip `put_rot`, through
  `ways` at the lift's height."""
  piece: str
  side: str
  pick: np.ndarray
  put: np.ndarray
  rot: np.ndarray
  put_rot: np.ndarray
  ways: list = field(default_factory=list)


def _grip(piece: str, middle, side: str, lev: Lever, tilt: float,
          rim: float | None = None) -> tuple[np.ndarray, np.ndarray]:
  """Where `side` grips `piece` lying with its middle at `middle` (start
  frame), and the grip: the plate at its wall `lev.rim` deg round from
  straight ahead (its left side 90, as the left hand takes it; the right
  hand its mirror), the jaws across the wall; the fork along its handle,
  the jaws across it; the cup across its body. The fingers lean ahead. The
  grip's height is set by the pick-up's tilt (`lev.tilt`), the tips where
  our procedure's 45 deg grip puts them: the grip point is the fingers'
  middle, half their length over the tips, so at 30 deg and 0.767 m they
  would only touch the plate's wall's top -- and held, it stays put on the
  piece whatever the hand turns to on its way."""
  x, y = middle
  half = hp.GRIPPER.finger_len / 2.0
  z = GRIP_Z[piece] + half * (math.sin(math.radians(lev.tilt)) - math.sin(math.radians(45.0)))
  sign = 1.0 if side == "left" else -1.0
  if piece == "plate":
    a = math.radians(lev.rim if rim is None else rim)
    u = (math.cos(a), sign * math.sin(a))
    at = (x + PLATE_GRIP_M * u[0], y + PLATE_GRIP_M * u[1], z)
    jaws = (u[0] * lev.jaws, u[1] * lev.jaws)
    lean = (u[1], -u[0]) if u[1] * sign >= 0 else (-u[1], u[0])
    if lean[0] < 0.0:
      lean = (-lean[0], -lean[1])
  else:
    at = (x + (lev.along if piece == "fork" else 0.0), y, z)
    jaws, lean = (0.0, float(lev.jaws)), (1.0, 0.0)
  # ⚠ our procedure's grip is `ws.GRIPS["t45"]` exactly: the jaws along y
  return np.array(at), grip_turn(tilt, jaws, lean)


def carries(lev: Lever, piece: str) -> list[Carry]:
  """`lev`'s carries of `piece`, start frame: one, or a relay's two."""
  side = lev.hand or HAND[piece]
  middle = _start(LAYOUTS[lev.layout][piece])
  spot = SPOT[piece]
  legs = [(middle, spot, side)]
  if lev.relay is not None:
    legs = [(middle, lev.relay, side), (lev.relay, spot, lev.relay_hand or side)]
  out = []
  for a, b, s in legs:
    pick, rot = _grip(piece, a, s, lev, lev.tilt)
    put, put_rot = _grip(piece, b, s, lev, lev.put_tilt if lev.put_tilt is not None else lev.tilt,
                         lev.put_rim)
    out.append(Carry(piece, s, pick, put, rot, put_rot))
  return out


@dataclass(frozen=True)
class Stance:
  """Where the stance stands: turned `turn` deg left, then `dx` ahead and
  `dy` left in the frame it then stands in (`step`'s order)."""
  dx: float = 0.0
  dy: float = 0.0
  turn: float = 0.0

  def point(self, p) -> np.ndarray:
    # rounded as `ws.window_tasks` rounds its stances' points
    x, y = _turned((float(p[0]), float(p[1])), -self.turn)
    return np.array([round(x - self.dx, 4), round(y - self.dy, 4), float(p[2])])

  def rot(self, r) -> np.ndarray:
    c, s = math.cos(math.radians(-self.turn)), math.sin(math.radians(-self.turn))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.asarray(r)

  def slab(self) -> tuple[np.ndarray, float]:
    """The table's top's edge in this stance's frame: the table where
    n . (x, y) > c."""
    th = math.radians(self.turn)
    n = np.array([math.cos(th), -math.sin(th)])
    return n, round(ct.TABLE_GAP - (math.cos(th) * self.dx - math.sin(th) * self.dy), 6)


def into_slab(sm, slab, top: float = ws.TABLE_H, thick: float = ws.TABLE_TOP_T) -> float:
  """`ws.into_table` with the table's edge anywhere in the stance's frame
  (`Stance.slab`): how far the robot's solid parts go into the top, m."""
  n, c = slab
  m, d = sm.m, sm.d
  worst = 0.0
  for g in range(m.ngeom):
    if m.geom_contype[g] == 0 or m.geom_bodyid[g] == 0:
      continue
    pts, r = ws._samples(m, d, g)
    inside = (pts[:, :2] @ n + r > c) & (pts[:, 2] - r < top) & (pts[:, 2] + r > top - thick)
    if inside.any():
      worst = max(worst, float((top - (pts[inside, 2] - r)).max()))
  return worst


def costs(sm, side, p, q_goal, q_pre, pos, rot, slab) -> dict:
  """`ws.costs` with the table's edge anywhere (`into_slab`): a planned
  reach's statue balanced again with the arm out, its clearances and its
  reach."""
  out = {}
  arm_goal = rc._arm(side, q_goal)
  sm.set_waist(p.waist)
  legs, ok = ws.static_legs(sm, p.height, p.pitch, arms=arm_goal)
  q = p.q.copy()
  for n, v in {**arm_goal, **legs}.items():
    q[hm.JOINT_NAMES.index(n)] = v
  sm.pose(q, ws.gravity_of(p.pitch))
  out["balanced"] = bool(ok)
  out["into_goal"] = round(into_slab(sm, slab), 4)
  out["self_goal"] = round(ws.self_clear(sm, side), 4)
  _, ep, er = rc.solve_seeded(sm, side, pos, rot, q_goal)
  out["still_reaches"] = bool(ep < ha.REACH_POS_TOL and er < ha.REACH_ROT_TOL)
  sm._set_arm(side, q_pre)
  out["into_pre"] = round(into_slab(sm, slab), 4)
  out["self_pre"] = round(ws.self_clear(sm, side), 4)
  return out


def _slerp_turn(r0: np.ndarray, r1: np.ndarray, share: float) -> np.ndarray:
  """`share` of the way from `r0` to `r1` about the axis between them."""
  e = ha._rot_error(r1, r0)
  a = float(np.linalg.norm(e))
  if a < 1e-9 or share >= 1.0:
    return np.array(r1, dtype=float)
  k = e / a
  t = a * max(0.0, share)
  kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
  return (np.eye(3) + math.sin(t) * kx + (1.0 - math.cos(t)) * kx @ kx) @ r0


def line(a, b, ra, rb, step: float = ws.CARRY_STEP_M) -> list[tuple[np.ndarray, np.ndarray]]:
  """A hand's line from `a` to `b`, its grip turning from `ra` to `rb` as a
  carried command turns (`Reach._line`): at `reach.HAND_TURN` while it
  walks at `reach.HOLD_SPEED` -- the turn done in its first centimetres."""
  a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
  n = max(1, math.ceil(float(np.linalg.norm(b - a)) / step))
  angle = float(np.linalg.norm(ha._rot_error(rb, ra)))
  turn_m = rc.HOLD_SPEED * angle / rc.HAND_TURN if angle > 1e-9 else 0.0
  out = []
  for k in range(1, n + 1):
    s = float(np.linalg.norm(b - a)) * k / n
    out.append((a + (b - a) * (k / n), _slerp_turn(ra, rb, 1.0 if turn_m <= 0.0 else s / turn_m)))
  return out


#: Our procedure's set-down: a `pull` down this far from 5 cm over the
#: put-down, m -- its way's end 4 cm into the table, whatever the lift --
#: held fast once the piece meets the table (`hands_check.TABLE`).
SET_DOWN_M = 0.09


def carry_points(c: Carry, lev: Lever, rot):
  """A carry's MOVES, each a list of (point, grip) on from where the move
  before left the hand: the lift (`pull` up, its turn held) from the pick-up
  to `a`, the carry (`move_hand`: straight, or through its way points) on to
  over the put-down, the set-down (`pull` down) -- and `a`, where the carry
  begins (#620's measure starts its carry there, `ws.carry_points`) -- and
  what the served reach plans each move on: the lift's way's end, the
  carry's way points and end, and the set-down's way's end, 9 cm down from
  over the put-down as our procedure's `pull` asks (`SET_DOWN_M`). `rot` is
  the grip the pick-up took (a flipped one carries its flip to the
  put-down's). Returns (a, [lift, carry, set-down], [their plans])."""
  flipped = not np.allclose(rot, c.rot)
  put_rot = c.put_rot @ np.diag([-1.0, -1.0, 1.0]) if flipped else c.put_rot
  up = np.array([0.0, 0.0, lev.lift])
  a, b = c.pick + up, c.put + up
  ways = []
  if lev.path == "round":
    mid = (a + b) / 2.0
    ways = [mid + np.array([lev.out, 0.0, 0.0])]
  elif lev.path == "ahead":
    ways = [b + np.array([lev.out, 0.0, 0.0])]
  lift = line(c.pick, a, rot, rot)
  carry, here, r, plan = [], a, rot, []
  for w in ways + [b]:
    # a way point keeps the turn it has; the end turns to the put-down's
    target = put_rot if w is b else r
    carry += line(here, w, r, target)
    plan.append((w, target))
    here, r = w, target
  down = line(b, c.put, put_rot, put_rot)
  below = c.put + np.array([0.0, 0.0, ws.LIFT_M - SET_DOWN_M])
  return a, [lift, carry, down], [[(a, rot)], plan, [(below, put_rot)]]


# ---- the three measures of a carry -------------------------------------------


def _reaches(ep: float, er: float, tol=ws.STRICT) -> bool:
  return ep < tol[0] and er < tol[1]


def solve_first(sm, side, pos, rot):
  """`reach.solve_seeded` from its seeds alone (no `start`), stopped at the
  first seed that reaches: with no start every seed within tolerance ranks
  alike, so the first is the one it keeps -- the same answer, sooner."""
  best, best_key = None, None
  for s_ in rc.SEEDS:
    sm.q[sm.arm[side]] = rc.seed(side, s_)
    q, ep, er = sm.solve(side, pos, rot, iters=80)
    if _reaches(ep, er):
      return q, ep, er
    key = ep + 0.1 * er
    if best_key is None or key < best_key:
      best, best_key = (q, ep, er), key
  return best


def follow_one(sm, side, order, pts, reach_m, slab):
  """`ws.carry_plan` (the waist held), each point its own grip: the least
  bent posture (the pick-up's first, in `order`) from which the grip
  follows `pts` [(point, grip)] in turn, the first solved from the seeds
  and each next on from the last, within the reach's own tolerance -- and
  its clearances on the way; or None. ⚠ The first that FOLLOWS is taken,
  clear or not, as the served planner (blind to the table) takes it."""
  for p in order:
    if not p.ok or any(np.linalg.norm(x - p.shoulder[side]) > reach_m + 0.005 for x, _ in pts):
      continue
    ws.pose(sm, p)
    q, ep, er = solve_first(sm, side, pts[0][0], pts[0][1])
    ok = _reaches(ep, er)
    worst_self, worst_into = 1.0, 0.0
    for x, r in (pts[1:] if ok else ()):
      sm._set_arm(side, q)
      q, ep, er = sm.solve(side, x, r, iters=80)
      if not _reaches(ep, er):
        ok = False
        break
      worst_self = min(worst_self, ws.self_clear(sm, side))
      worst_into = max(worst_into, into_slab(sm, slab))
    if ok:
      return p, q, worst_self, worst_into
  return None


def repose(sm, side, table, start_label, pts, reach_m, slab) -> dict:
  """`ws.reposture_carry` over `pts` [(point, grip)]: a search over (point,
  posture), on along a posture each next point solved on from the last
  within the served carry's tolerance (`ws.LOOSE`; the last within the
  reach's own), or another posture with the hand held still where both
  reach the point, every point clear. Starts at the pick-up's posture."""
  n = len(pts)

  def clear():
    return ws.self_clear(sm, side) >= -ws.SELF_M and into_slab(sm, slab) <= ws.INTO_M

  memo: dict = {}

  def seeded(k: int, i: int):
    """Posture i's own solve of point k from the seeds, clear -- or None:
    solved only once the search reaches point k (the same search, sooner:
    #620's solved every point of every posture first)."""
    if (k, i) not in memo:
      memo[(k, i)] = None
      x, r = pts[k]
      if np.linalg.norm(x - table[i].shoulder[side]) <= reach_m + 0.005:
        ws.pose(sm, table[i])
        q, ep, er = solve_first(sm, side, x, r)
        tol = ws.STRICT if k == n - 1 else ws.LOOSE
        sm._set_arm(side, q)
        if ep < tol[0] and er < tol[1] and clear():
          memo[(k, i)] = q
    return memo[(k, i)]

  start = [(0, i) for i, p in enumerate(table) if p.label() == start_label and seeded(0, i) is not None]
  arm = {node: seeded(*node) for node in start}
  seen, todo, prev = set(start), deque(start), {}
  end = None
  while todo:
    k, i = todo.popleft()
    if k == n - 1:
      end = (k, i)
      break
    if (k + 1, i) not in seen and np.linalg.norm(pts[k + 1][0] - table[i].shoulder[side]) <= reach_m + 0.005:
      ws.pose(sm, table[i])
      sm._set_arm(side, arm[(k, i)])
      q, ep, er = sm.solve(side, pts[k + 1][0], pts[k + 1][1], iters=80)
      tol = ws.STRICT if k + 1 == n - 1 else ws.LOOSE
      if ep < tol[0] and er < tol[1] and clear():
        seen.add((k + 1, i))
        prev[(k + 1, i)] = (k, i)
        arm[(k + 1, i)] = q
        todo.append((k + 1, i))
    for j in range(len(table)):
      if (k, j) not in seen and seeded(k, j) is not None:
        seen.add((k, j))
        prev[(k, j)] = (k, i)
        arm[(k, j)] = seeded(k, j)
        todo.append((k, j))
  out = {"carried": end is not None, "furthest": max((k for k, _ in seen), default=-1),
         "points": n}
  if end is not None:
    labels, node = [], end
    while True:
      labels.append(table[node[1]].label())
      if node not in prev:
        break
      node = prev[node]
    labels = labels[::-1]
    out["postures"] = [lab for k, lab in enumerate(labels) if k == 0 or lab != labels[k - 1]]
  return out


def _hold_first(hold, here_label):
  return [p for p in hold if p.label() == here_label] + [p for p in hold if p.label() != here_label]


def _balanced(sm, side, p, q) -> None:
  """Posture `p`'s statue balanced again with `side`'s arm at `q`."""
  sm.set_waist(p.waist)
  legs, _ = ws.static_legs(sm, p.height, p.pitch, arms=rc._arm(side, q))
  qq = p.q.copy()
  for n_, v in legs.items():
    qq[hm.JOINT_NAMES.index(n_)] = v
  sm.pose(qq, ws.gravity_of(p.pitch))


def path_for(sm, side, hold, here_label, q_here, pts, reach_m):
  """`Reach.path_for` on the statue, BLIND to the table as the served
  planner is: the posture the hand is in first (its arm as it is, the
  statue balanced with it out: the live branch), then the least bent of
  `hold`, from which the grip follows `pts` [(point, grip)] from the hand's
  place, each next on from the last -- in the posture it holds, a point
  further from its shoulder than the first reached `LIVE_MARGIN_M` further
  ahead; in another, every point after the first reached `PATH_MARGIN_M`
  further ahead on a statue balanced with the arm out at the point before
  -- within `PATH_POS_TOL` and `HOLD_ROT_TOL`. The posture, or None."""
  ahead = np.array([rc.PATH_MARGIN_M, 0.0, 0.0])
  live_ahead = np.array([rc.LIVE_MARGIN_M, 0.0, 0.0])

  def within(ep, er):
    return ep < rc.PATH_POS_TOL and er < rc.HOLD_ROT_TOL

  for p in _hold_first(hold, here_label):
    if not p.ok or any(np.linalg.norm(x - p.shoulder[side]) > reach_m + 0.005 for x, _ in pts):
      continue
    live = p.label() == here_label and q_here is not None
    if live:
      _balanced(sm, side, p, q_here)
      q = np.array(q_here, dtype=float)
      shoulder = sm.d.xpos[sm.m.body(f"{side}_shoulder_roll_link").id].copy()
      reach0 = float(np.linalg.norm(pts[0][0] - shoulder))
    else:
      ws.pose(sm, p)
      q, ep, er = rc.solve_seeded(sm, side, pts[0][0], pts[0][1], q_here)
      if not within(ep, er):
        continue
    ok = True
    for x, r in pts[1:]:
      if live:
        target = x + live_ahead if float(np.linalg.norm(x - shoulder)) > reach0 else x
      else:
        _balanced(sm, side, p, q)
        target = x + ahead
      sm._set_arm(side, q)
      q, ep, er = sm.solve(side, target, r, iters=80)
      if not within(ep, er):
        ok = False
        break
    if ok:
      return p
  return None


def posture_for(sm, side, hold, here_label, q_cmd, pos, rot, reach_m):
  """`Reach.posture_for` on the statue: the posture held first, then the
  least bent of `hold`, from which the grip reaches `pos` at `rot` within
  the reach's own tolerance, from the arm as commanded or from the seeds;
  or None."""
  for p in _hold_first(hold, here_label):
    if not p.ok or np.linalg.norm(pos - p.shoulder[side]) > reach_m + 0.005:
      continue
    ws.pose(sm, p)
    for start in [q_cmd] + [rc.seed(side, s_) for s_ in rc.SEEDS]:
      sm._set_arm(side, start)
      _, ep, er = sm.solve(side, pos, rot, iters=80)
      if _reaches(ep, er):
        return p
  return None


def posture_onward(sm, side, hold, gone, q_cmd, pos, rot, reach_m):
  """`Reach.posture_for(onward=True)` on the statue, as a pull takes
  another posture where its arm misses on its way: the least bent of `hold`
  but those in `gone` (the one it holds, and those it has left this move),
  from which the grip reaches `pos` at `rot` within the reach's own
  tolerance, from the arm as commanded alone; or None."""
  for p in hold:
    if not p.ok or p.label() in gone or np.linalg.norm(pos - p.shoulder[side]) > reach_m + 0.005:
      continue
    ws.pose(sm, p)
    sm._set_arm(side, q_cmd)
    _, ep, er = sm.solve(side, pos, rot, iters=80)
    if _reaches(ep, er):
      return p
  return None


def served(sm, side, hold, p_pick, q_pick, pick, taken, moves, plan_pts, fixed, reach_m, slab):
  """Our procedure's carry as the served reach makes it, on the statue,
  every posture one of the carry's own (`hold`), from the pick-up's
  posture and arm on:
  - each `pull` (the lift; the set-down, its way 9 cm down, held fast on
    the table) takes the posture `path_for` plans from the hand to its
    way's end, the hand held still while the body comes, and where its arm
    misses on its way another, onward (`posture_onward`);
  - `move_hand` takes the posture `posture_for` finds for its end and walks
    its line, a miss past `REPLAN_M` or `REPLAN_ROT` taking the least bent
    posture that reaches where it is (`posture_for`) with the hand held
    still -- or, through way points or turning (`fixed`), the one posture
    `path_for` plans for them all; a posture planned at most once a second
    of the line (`REPLAN_EVERY`);
  - each point solved on from the last, the arm continuing from where it
    is commanded (a posture taken mid-move keeps the arm's branch); a miss
    past `BLOCKED_M` or `HOLD_ROT_TOL` is out of reach; a carry arrives
    within `ARRIVE_M`, a set-down within the reach's own tolerance; every
    point clear of the table and of the robot's own parts.
  `plan_pts` are what each move is planned on (the pulls' ends, the carry's
  way points and end). Returns (ok, why, the postures taken)."""
  here, q = p_pick, np.array(q_pick, dtype=float)
  taken_postures = [here.label()]
  at = (pick, taken)

  def take(p):
    nonlocal here
    if p is not None and p.label() != here.label():
      here = p
      taken_postures.append(p.label())

  for k, mv in enumerate(moves):
    name = ("lift", "carry", "set-down")[k]
    if name == "carry" and not fixed:
      take(posture_for(sm, side, hold, here.label(), q, mv[-1][0], mv[-1][1], reach_m))
    else:
      take(path_for(sm, side, hold, here.label(), q, [at] + plan_pts[k], reach_m))
    ws.pose(sm, here)
    ep, planned, left = 0.0, -REPLAN_EVERY, []
    for j, (x, r) in enumerate(mv):
      sm._set_arm(side, q)
      q2, ep, er = sm.solve(side, x, r, iters=80)
      if ((ep > rc.REPLAN_M or er > rc.REPLAN_ROT) and not (name == "carry" and fixed)
          and j - planned >= REPLAN_EVERY):
        # ...at most once a second, as the served carry plans one: a
        # straight carry the least bent that reaches where it is (it may go
        # back to one it left), a pull another onward
        planned = j
        if name == "carry":
          p = posture_for(sm, side, hold, here.label(), q, x, r, reach_m)
        else:
          p = posture_onward(sm, side, hold, {here.label(), *left}, q, x, r, reach_m)
          if p is not None:
            left.append(here.label())
        if p is not None and p.label() != here.label():
          take(p)
          ws.pose(sm, here)
          sm._set_arm(side, q)
          q2, ep, er = sm.solve(side, x, r, iters=80)
      if ep > rc.BLOCKED_M or er > rc.HOLD_ROT_TOL:
        return False, f"{name}: out of reach from {here.label()}", taken_postures
      if ep < rc.WILD_M:
        q = q2
      sm._set_arm(side, q)
      if ws.self_clear(sm, side) < -ws.SELF_M or into_slab(sm, slab) > ws.INTO_M:
        return False, f"{name}: not clear from {here.label()}", taken_postures
    if name == "carry" and ep > rc.ARRIVE_M:
      return False, f"carry: held back {100 * ep:.1f} cm short", taken_postures
    at = mv[-1]
  if ep >= ha.REACH_POS_TOL:
    return False, f"set-down: {100 * ep:.1f} cm short of the table", taken_postures
  return True, "", taken_postures


# ---- one task: a carry from one stance ----------------------------------------

_HOLD: dict = {}


#: Where #620's search over postures (`repose`) is run: today's body, where
#: it agrees with the flights at our procedure's plate step. On a waist the
#: served carry measures posture changes (`served`): the search tries every
#: posture at every point, 36 s of a yaw stance's 57.
REPOSE_KINDS = ("none",)
#: A straight carry takes another posture at most once in this many points
#: of its line: `reach.REPLAN_S` at `HOLD_SPEED`, `CARRY_STEP_M` apart.
REPLAN_EVERY = max(1, round(rc.REPLAN_S * rc.HOLD_SPEED / ws.CARRY_STEP_M))


def _hold(kind: str):
  """A candidate's carry postures: `reach.HOLD_POSTURES` with each of its
  waist's settings (`ws.postures`), as a carried hand may take them."""
  if kind not in _HOLD:
    sm, _, _ = ws._models(kind)
    _HOLD[kind] = ws.postures(sm, legs=rc.HOLD_POSTURES)
  return _HOLD[kind]


def evaluate(task: dict) -> dict:
  """A lever's carry `task["leg"]` of `task["piece"]` from the stance
  (`task["dx"]`, `["dy"]`, `["turn"]`): the pick-up planned as `ws.plan`
  plans it and judged as `ws._ok` judges it, then the carry by each of the
  lever's measures."""
  lev = LEVERS[task["lever"]]
  kind = task["kind"]
  c = carries(lev, task["piece"])[task.get("leg", 0)]
  st = Stance(task["dx"], task["dy"], task["turn"])
  sm, table, reach_m = ws._models(kind)
  slab = st.slab()
  pick, rot = st.point(c.pick), st.rot(c.rot)
  c_st = Carry(c.piece, c.side, pick, st.point(c.put), rot, st.rot(c.put_rot))
  out = dict(task, side=c.side, pick=[round(float(v), 4) for v in pick],
             put=[round(float(v), 4) for v in c_st.put], reached=False)
  t0 = time.perf_counter()
  got = ws.plan(sm, c.side, ha.HandGoal(pick, rot), table, reach_m, flip=lev.flip)
  if got is None:
    out["s"] = round(time.perf_counter() - t0, 3)
    return out
  p, q_goal, slack, q_pre, taken = got
  out.update(reached=True, slack=bool(slack), posture=p.label(),
             flipped=bool(not np.allclose(taken, rot)))
  out.update(costs(sm, c.side, p, q_goal, q_pre, pick, taken, slab))
  out["pick_ok"] = ws._ok(out)
  if not out["pick_ok"]:
    out["s"] = round(time.perf_counter() - t0, 3)
    return out
  a, moves, plans = carry_points(c_st, lev, taken)
  models = task.get("models") or lev.models
  # `one` and `repose` from the lift's top on, as #620 measures a carry
  pts = [(a, taken)] + moves[1] + moves[2]
  order = [q for q in table if q.label() == p.label()] + [q for q in table if q.label() != p.label()]
  if "one" in models:
    got1 = follow_one(sm, c.side, order, pts, reach_m, slab)
    out["one"] = bool(got1 is not None and got1[3] <= ws.INTO_M and got1[2] >= -ws.SELF_M)
    if got1 is not None:
      out["one_posture"], out["one_into"], out["one_self"] = (got1[0].label(), round(got1[3], 4),
                                                              round(got1[2], 4))
  if "repose" in models and "repose" in lev.models and kind in REPOSE_KINDS:
    if out.get("one"):
      out["repose"] = True
    else:
      r = repose(sm, c.side, [q for q in table if q.ok], p.label(), pts, reach_m, slab)
      out["repose"] = r["carried"]
      out["repose_furthest"] = r["furthest"]
      if r.get("postures"):
        out["repose_postures"] = r["postures"]
  if "served" in models and "served" in lev.models:
    # the lift, the carry and the set-down as the served reach makes them,
    # from the pick-up's posture -- the carry's own postures among them
    hold = _hold(kind)
    p_hold = next((q for q in hold if q.label() == p.label()), p)
    fixed = lev.path != "straight" or lev.put_tilt is not None or lev.put_rim is not None
    ok, why, postures = served(sm, c.side, hold, p_hold, q_goal, pick, taken, moves, plans, fixed,
                               reach_m, slab)
    out["served"] = ok
    out["served_postures"] = postures
    if why:
      out["served_why"] = why
  out["s"] = round(time.perf_counter() - t0, 3)
  return out


def _evaluate(task: dict) -> dict:
  try:
    return evaluate(task)
  except Exception as e:      # a row says what went wrong, never the batch
    return dict(task, error=repr(e))


# ---- the sweeps -------------------------------------------------------------------

#: The stances: across from 0.50 m to the right to 0.50 to the left of the
#: start, 1 cm apart (`ws.WINDOW_DY`'s, wider: the other hand and a turned
#: stance stand further over).
DY = tuple(round(-0.50 + 0.01 * i, 2) for i in range(101))


def tasks_for(levers, kinds, pieces=None, models=None) -> list[dict]:
  """Every stance of `levers` on `kinds` (their `pieces`, or those named),
  measured by the levers' own measures, or by `models` alone."""
  out = []
  for k in kinds:
    for name in levers:
      lev = LEVERS[name]
      for piece in lev.pieces:
        if pieces and piece not in pieces:
          continue
        for leg in range(len(carries(lev, piece))):
          for dy in DY:
            out.append({"kind": k, "lever": name, "piece": piece, "leg": leg,
                        "dx": lev.dx, "dy": dy, "turn": lev.turn,
                        **({"models": list(models)} if models else {})})
  return out


def run(tasks: list[dict], jobs: int, into: str | None) -> list[dict]:
  """`ws.run` for these tasks: each row to `into`/rows.jsonl as it comes,
  one kind's together (a process builds a kind's postures once)."""
  import json
  import os
  tasks = sorted(tasks, key=lambda t: (KINDS.index(t["kind"]) if t["kind"] in KINDS else 9, t["lever"]))
  fh = None
  if into:
    os.makedirs(into, exist_ok=True)
    path = os.path.join(into, "rows.jsonl")
    if os.path.exists(path):
      sys.exit(f"{path} exists: --into a directory of its own")
    fh = open(path, "w")
  rows, t0 = [], time.perf_counter()
  pool = mp.get_context("spawn").Pool(jobs) if jobs > 1 else None
  try:
    it = pool.imap(_evaluate, tasks, chunksize=4) if pool else map(_evaluate, tasks)
    for k, row in enumerate(it, 1):
      rows.append(row)
      if fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
      if k % 200 == 0 or k == len(tasks):
        print(f"  {k}/{len(tasks)} rows, {time.perf_counter() - t0:.0f} s", flush=True)
  finally:
    if pool:
      pool.terminate()
      pool.join()
    if fh:
      fh.close()
  return rows


# ---- the reports -----------------------------------------------------------------

#: The bar (module docstring): a run of this many good stances, 1 cm apart.
BAR_CM = 6


def good(r: dict, model: str) -> bool:
  return bool(r.get("pick_ok")) and bool(r.get(model))


def runs(dys) -> list[tuple[float, float]]:
  return ws._intervals(sorted(dys), 0.01)


def widest(iv) -> int:
  return max((round(100 * (b - a)) + 1 for a, b in iv), default=0)


def window(rows, kind, lever, piece, leg, model) -> list[tuple[float, float]]:
  return runs([r["dy"] for r in rows if r["kind"] == kind and r["lever"] == lever
               and r["piece"] == piece and r.get("leg", 0) == leg and good(r, model)])


def measured(kind: str, model: str) -> bool:
  """Whether `model` is measured on `kind` (`REPOSE_KINDS`)."""
  return model != "repose" or kind in REPOSE_KINDS


def clears(rows, kind, name, model) -> dict:
  """Each of lever `name`'s pieces (a relay's legs each) on `kind` by
  `model`: its widest window, cm, and where -- or None where the lever
  does not measure it by `model`."""
  lev = LEVERS[name]
  if model not in lev.models or not measured(kind, model):
    return {}
  out = {}
  for piece in lev.pieces:
    legs = sorted({r.get("leg", 0) for r in rows if r["lever"] == name and r["piece"] == piece})
    for leg in legs:
      iv = window(rows, kind, name, piece, leg, model)
      best = max(iv, key=lambda ab: ab[1] - ab[0], default=None)
      out[piece if len(legs) == 1 else f"{piece}.{leg + 1}"] = (widest(iv), best)
  return out


def report_bar(rows) -> None:
  """The bar (`BAR_CM`) lever by lever: which clear it for the plate and
  the fork (a relay's every leg), by each measure."""
  kinds = [k for k in KINDS if any(r["kind"] == k for r in rows)]
  levers = [n for n in LEVERS if any(r["lever"] == n for r in rows)]
  print(f"\nTHE BAR: a window of at least {BAR_CM} cm for the plate and for the fork (every "
        "leg of a relay), by each measure: the levers that clear it, and how wide")
  for kind in kinds:
    print(f"\n== {kind}")
    for model in [m for m in ("one", "repose", "served") if measured(kind, m)]:
      got = []
      for name in levers:
        c = clears(rows, kind, name, model)
        need = [k for k in c if k.split(".")[0] in ("plate", "fork")]
        if need and all(c[k][0] >= BAR_CM for k in need):
          got.append(f"{name} (" + ", ".join(f"{k} {c[k][0]}" for k in need) + ")")
      print(f"  {model:7s} " + ("; ".join(got) if got else "none"))
    for piece in ("plate", "fork"):
      for model in [m for m in ("one", "repose", "served") if measured(kind, m)]:
        best = []
        for name in levers:
          c = clears(rows, kind, name, model)
          for k, (w, ab) in c.items():
            if k.split(".")[0] == piece and w >= BAR_CM:
              best.append((w, f"{name}{k[len(piece):]} {w} ({ab[0]:+.2f}..{ab[1]:+.2f})"))
        best.sort(key=lambda b: -b[0])
        print(f"  {piece:5s} {model:7s} " + ("; ".join(b[1] for b in best[:12]) if best else "none")
              + (f"; and {len(best) - 12} more" if len(best) > 12 else ""))


def report(rows) -> None:
  errs = [r for r in rows if "error" in r]
  if errs:
    print(f"⚠ {len(errs)} rows raised; the first: {errs[0]['error']}")
  kinds = [k for k in KINDS if any(r["kind"] == k for r in rows)]
  levers = [n for n in LEVERS if any(r["lever"] == n for r in rows)]
  print("\nTHE PUT-DOWNS' WINDOWS: each lever's stances (m across off the start; dx, turn as the "
        "lever stands) from which the piece is picked up and carried where it goes and set "
        f"down, clear: the widest run, cm (the bar {BAR_CM}), by `one` posture / `repose`d on "
        "the way / `served` as the served reach makes it; then the runs")
  for kind in kinds:
    print(f"\n== {kind}")
    for name in levers:
      lev = LEVERS[name]
      cells = []
      for piece in lev.pieces:
        legs = sorted({r.get("leg", 0) for r in rows if r["lever"] == name and r["piece"] == piece})
        for leg in legs:
          ws_ = []
          for model in ("one", "repose", "served"):
            if model in lev.models:
              ws_.append(f"{widest(window(rows, kind, name, piece, leg, model)):2d}"
                         if measured(kind, model) else " -")
          tag = piece if len(legs) == 1 else f"{piece}.{leg + 1}"
          cells.append(f"{tag} {'/'.join(ws_)}")
      print(f"  {name:18s} " + "   ".join(cells))
  print("\nTHE RUNS, by `one` posture (and `repose`d where it adds any):")
  for kind in kinds:
    print(f"\n== {kind}")
    for name in levers:
      lev = LEVERS[name]
      for piece in lev.pieces:
        legs = sorted({r.get("leg", 0) for r in rows if r["lever"] == name and r["piece"] == piece})
        for leg in legs:
          one = window(rows, kind, name, piece, leg, "one")
          line_ = f"  {name:18s} {piece}{'' if len(legs) == 1 else f'.{leg + 1}'}: {ws._fmt(one)}"
          if "repose" in lev.models and measured(kind, "repose"):
            rp = window(rows, kind, name, piece, leg, "repose")
            if rp != one:
              line_ += f"; repose {ws._fmt(rp)}"
          if "served" in lev.models:
            line_ += f"; served {ws._fmt(window(rows, kind, name, piece, leg, 'served'))}"
          print(line_)
  report_bar(rows)


def check(windows: bool = True) -> None:
  """--check: the base lever's rows against #620's window record, cell by
  cell -- the instrument extended, its calibration kept (`ws.evaluate_piece`
  at the same stances)."""
  bad = 0
  for kind in (("none", "yaw") if windows else ()):
    for piece in ("plate", "fork", "cup"):
      side, pick, put, step = ws.PIECES[piece]
      for dy in (-0.31, -0.29, -0.26, -0.21, -0.12, -0.09, 0.0, 0.04):
        task = {"sweep": "window", "kind": kind, "piece": piece, "side": side, "grip": "t45",
                "dx": 0.0, "dy": dy, "edge": round(ws.TABLE_GAP, 4), "which": "pick",
                "pos": [pick[0], round(pick[1] - dy, 4), pick[2]],
                "put": [put[0], round(put[1] - dy, 4), put[2]]}
        old = ws.evaluate_piece(task)
        old_ok = ws._ok(old) and old.get("carried", False) and old.get("into_carried", 0.0) <= ws.INTO_M \
          and old.get("self_carried", 1.0) >= -ws.SELF_M
        new = evaluate({"kind": kind, "lever": "base", "piece": piece, "leg": 0, "dx": 0.0,
                        "dy": dy, "turn": 0.0})
        same = (ws._ok(old) == new.get("pick_ok", False)) and (old_ok == good(new, "one"))
        bad += not same
        print(f"  {kind:5s} {piece:5s} dy {dy:+.2f}: #620 picked {ws._ok(old)!s:5s} carried "
              f"{old_ok!s:5s} | here picked {new.get('pick_ok', False)!s:5s} carried "
              f"{good(new, 'one')!s:5s} {'' if same else '  <-- DIFFERS'}", flush=True)
  # ...and its search over postures is #620's `--reposture`, cell by cell
  for kind in ("none", "yaw"):
    for piece in ("plate", "fork"):
      for dy in (-0.26, -0.20, -0.09):
        old = ws.reposture_carry(kind, piece, dy)
        sm, table, reach_m = ws._models(kind)
        side, pick, put, _ = ws.PIECES[piece]
        pick = np.array(pick) - (0.0, dy, 0.0)
        put = np.array(put) - (0.0, dy, 0.0)
        got = ws.plan(sm, side, ha.HandGoal(pick, ws.GRIPS["t45"]), [q for q in table if q.ok], reach_m)
        if got is None:
          same = not old["picked"]
          bad += not same
          print(f"  {kind:5s} {piece:5s} dy {dy:+.2f}: not picked up here, #620 "
                f"{'nor' if same else 'DID'}", flush=True)
          continue
        pts = [(x, got[4]) for x in ws.carry_points(pick, put)]
        new = repose(sm, side, [q for q in table if q.ok], got[0].label(), pts, reach_m,
                     Stance().slab())
        same = (old["carried"] == new["carried"] and old["furthest"] == new["furthest"]
                and old.get("postures") == new.get("postures"))
        bad += not same
        print(f"  {kind:5s} {piece:5s} dy {dy:+.2f}: #620 reposture {old['carried']!s:5s} to point "
              f"{old['furthest']:2d} {old.get('postures', '')} | here {new['carried']!s:5s} to point "
              f"{new['furthest']:2d} {new.get('postures', '')}{'' if same else '  <-- DIFFERS'}",
              flush=True)
  print("the base lever is #620's window and its search" if not bad else f"⚠ {bad} cells differ")


def read_rows(into: str) -> list[dict]:
  """The rows of one record, or of several (`DIR,DIR`: a batch's shares)."""
  import json
  rows = []
  for d in into.split(","):
    with open(Path(d) / "rows.jsonl") as fh:
      rows += [json.loads(line_) for line_ in fh]
  return rows


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--levers", default="")
  ap.add_argument("--candidates", default=",".join(KINDS))
  ap.add_argument("--pieces", default="")
  ap.add_argument("--jobs", type=int, default=3)
  ap.add_argument("--into")
  ap.add_argument("--read", metavar="DIR")
  ap.add_argument("--sample", type=int, default=0)
  ap.add_argument("--part", default="", help="I/N: the Ith of N shares of the tasks")
  ap.add_argument("--models", default="", help="one,repose,served: these measures alone")
  ap.add_argument("--list", action="store_true")
  ap.add_argument("--check", action="store_true")
  args = ap.parse_args()
  if args.list:
    for lev in LEVERS.values():
      print(f"  {lev.name:18s} {lev.note}  [{', '.join(lev.pieces)}; {', '.join(lev.models)}]")
    return
  if args.check:
    check()
    return
  if args.read:
    report(read_rows(args.read))
    return
  levers = [n for n in args.levers.split(",") if n] or list(LEVERS)
  for n in levers:
    if n not in LEVERS:
      sys.exit(f"no lever {n!r}: --list")
  kinds = [k for k in args.candidates.split(",") if k]
  for k in kinds:
    if k not in ws.CANDIDATES:
      sys.exit(f"no candidate {k!r}: {', '.join(ws.CANDIDATES)}")
  pieces = [p for p in args.pieces.split(",") if p]
  models = [m for m in args.models.split(",") if m]
  for m in models:
    if m not in ("one", "repose", "served"):
      sys.exit(f"no measure {m!r}: one, repose, served")
  tasks = tasks_for(levers, kinds, pieces, models)
  if args.sample > 1:
    tasks = tasks[::args.sample]
  if args.part:
    i, n = (int(v) for v in args.part.split("/"))
    tasks = tasks[i - 1::n]
  print(f"{len(tasks)} stances to plan, {len(levers)} levers on {', '.join(kinds)}, "
        f"{args.jobs} processes", flush=True)
  rows = run(tasks, args.jobs, args.into)
  report(rows)


if __name__ == "__main__":
  main()
