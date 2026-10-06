"""The scene language (issue #466): what the robot writes to describe what is
in front of it, and what `compile.py` builds a world of its own from. On
`workshop/spec.py`'s pattern -- JSON, closed, every reason at once:

  {"parts": [
     {"id": "base", "shape": "box", "size": [220, 300, 140],
      "pos": [1060, 0, 70], "mass": 2.0},
     {"id": "lid", "shape": "slab", "size": [220, 300, 12],
      "pos": [1060, 0, 146], "on": "base", "mass": 0.3}],
   "joints": [
     {"id": "hinge", "type": "hinge", "part": "lid", "at": [1170, 0, 140],
      "axis": [0, 1, 0], "range": [0, 109], "stiffness": 0.02, "slack": 143,
      "damping": 0.04, "friction": 0.04}],
   "catches": [{"joint": "hinge", "at": [900, 0, 146], "release": 2.0}]}

A PART is one primitive with its own mass: a `box` or a `slab` (`size` its
x, y and z extents; a slab is a board, its z its thickness, at most a
quarter of the other two) or a `cylinder` (`size` its diameter and its
length, along its z). `pos` is its middle and `euler` its turn (MuJoCo's
x-y-z, default none), both in the robot's own MAP frame. `on` names the part
it rides; a part on nothing is fixed in the map.

A JOINT lets its `part` move against what that part is `on` (or the map):
a `hinge` turns about `axis` through `at`, a `slide` runs along `axis` (its
`at` only says where).
`range` is where it may go, off the pose the document describes (which
must lie in it; none is unlimited). A spring of `stiffness` is slack at
`slack`; `damping` and Coulomb `friction` are the joint's own.

A CATCH holds its `joint` at the end of its range where the document's
pose is -- a range [0, 109] is caught at 0 -- with one pull, `release`
newtons at `at`, until the part is pulled harder there: then it lets go.

  ⚠ UNITS ARE THE AUTHOR'S AT THE DOOR AND THE SIM'S INSIDE, converted once
  by `parse`: lengths in mm and angles in degrees (a slide's range and
  slack in mm); mass in kg; a hinge's stiffness, damping and friction in
  N*m/rad, N*m*s/rad and N*m (a slide's in N/m, N*s/m and N), as #466
  states a lid's; a release in N.

  ⚠ AN UNKNOWN FIELD IS REFUSED, NOT IGNORED. There is no field for a contact
  parameter, a solver option or an armature: the imagination's contact is
  its own (`compile.CONTACT`), and a document that could carry the world's
  would be a door for it (`tests/test_imagination.py` pins `solref`).

Nothing here is specific to a lid.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np

SHAPES = ("box", "slab", "cylinder")
JOINT_TYPES = ("hinge", "slide")
ID = re.compile(r"^[a-z][a-z0-9_]{0,23}$")

DOC_FIELDS = {"parts", "joints", "catches"}
PART_FIELDS = {"id", "shape", "size", "pos", "euler", "on", "mass"}
JOINT_FIELDS = {"id", "type", "part", "at", "axis", "range", "stiffness", "slack",
                "damping", "friction"}
CATCH_FIELDS = {"joint", "at", "release"}

#: What a document may hold: a scene in front of a robot, not a house.
MAX_PARTS, MAX_JOINTS, MAX_CATCHES = 24, 8, 4
#: A part's extent, mm: under a millimetre a contact is the solver's guess,
#: and over 3 m it is furniture the arm cannot reach round.
MIN_SIZE_MM, MAX_SIZE_MM = 1.0, 3000.0
#: Where anything may be, mm off the map's origin either way.
MAX_POS_MM = 100_000.0
#: A part's mass, kg.
MAX_MASS_KG = 200.0
#: A slab's thickness is at most this share of its other two extents.
SLAB_RATIO = 4.0
#: A hinge's range, degrees either way; a slide's is its size's.
MAX_HINGE_DEG = 360.0
#: A catch's pull, N.
MAX_RELEASE_N = 200.0
#: A catch on a hinge needs a lever: its point this far off the axis, mm.
MIN_LEVER_MM = 1.0
#: An axis is a direction, its components within this either way.
MAX_AXIS = 1e6


class Refused(ValueError):
  """A document that does not parse. `reasons` is every rule it broke, so an
  author fixes them together rather than one per refusal."""

  def __init__(self, reasons: list[str]) -> None:
    super().__init__("; ".join(reasons))
    self.reasons = list(reasons)


@dataclass(frozen=True)
class Part:
  """One primitive, in the sim's units: `size` m (a cylinder's is its
  diameter and its length), `pos` m and `euler` rad in the map frame."""
  id: str
  shape: str
  size: tuple[float, ...]
  pos: tuple[float, float, float]
  euler: tuple[float, float, float]
  on: str | None
  mass: float


@dataclass(frozen=True)
class Joint:
  """One joint, in the sim's units: `at` m, `axis` a unit vector, both in
  the map frame; `range` rad (hinge) or m (slide) off the document's pose,
  None unlimited; `slack` likewise."""
  id: str
  type: str
  part: str
  at: tuple[float, float, float]
  axis: tuple[float, float, float]
  range: tuple[float, float] | None
  stiffness: float
  slack: float
  damping: float
  friction: float


@dataclass(frozen=True)
class Catch:
  """A catch on `joint`, pulling `release` N at `at` (m, map frame)."""
  joint: str
  at: tuple[float, float, float]
  release: float


@dataclass(frozen=True)
class Scene:
  parts: tuple[Part, ...]
  joints: tuple[Joint, ...] = ()
  catches: tuple[Catch, ...] = ()

  def part(self, pid: str) -> Part:
    return next(p for p in self.parts if p.id == pid)

  def joint(self, jid: str) -> Joint:
    return next(j for j in self.joints if j.id == jid)


def rotation(euler_rad) -> np.ndarray:
  """MuJoCo's default euler sequence: intrinsic x-y-z, radians."""
  x, y, z = (float(v) for v in euler_rad)
  cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), \
    math.cos(z), math.sin(z)
  rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
  ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
  rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
  return rx @ ry @ rz


# ---- parsing ----------------------------------------------------------------------

def _finite(x) -> bool:
  """A JSON number that is a float's: an integer of 309 digits is none
  (`math.isfinite` raises on it)."""
  if not isinstance(x, (int, float)) or isinstance(x, bool):
    return False
  try:
    return math.isfinite(float(x))
  except OverflowError:
    return False


def _num(v, name: str, reasons: list[str], lo: float | None = None,
         hi: float | None = None) -> float | None:
  if not _finite(v):
    reasons.append(f"{name} must be a number")
    return None
  v = float(v)
  if lo is not None and v < lo:
    reasons.append(f"{name} is {v:g}; it may be no less than {lo:g}")
    return None
  if hi is not None and v > hi:
    reasons.append(f"{name} is {v:g}; it may be no more than {hi:g}")
    return None
  return v


def _nums(v, name: str, reasons: list[str], n: int, bound: float | None = None):
  if not isinstance(v, list) or len(v) != n or not all(_finite(x) for x in v):
    reasons.append(f"{name} must be {n} numbers")
    return None
  if bound is not None and any(abs(x) > bound for x in v):
    reasons.append(f"{name} must be within {bound:g} either way")
    return None
  return tuple(float(x) for x in v)


def _fields(raw: dict, allowed: set[str], where: str, reasons: list[str]) -> None:
  for key in raw:
    if key not in allowed:
      reasons.append(f"{where}: unknown field {key!r} (it may have "
                     f"{', '.join(sorted(allowed))})")


def _id(raw: dict, where: str, reasons: list[str]) -> str | None:
  v = raw.get("id")
  if isinstance(v, str) and ID.fullmatch(v):              # `match` lets a "\n" end it
    return v
  reasons.append(f"{where}: id must match {ID.pattern}")
  return None


def _part(raw, i: int, reasons: list[str]) -> Part | None:
  where = f"part {i}"
  if not isinstance(raw, dict):
    reasons.append(f"{where}: must be an object")
    return None
  pid = _id(raw, where, reasons)
  where = f"part {pid!r}" if pid else where
  _fields(raw, PART_FIELDS, where, reasons)
  shape = raw.get("shape")
  if shape not in SHAPES:
    reasons.append(f"{where}: shape must be one of {', '.join(SHAPES)}")
  n = 2 if shape == "cylinder" else 3
  size = _nums(raw.get("size"), f"{where}: size", reasons, n)
  if size is not None and any(not MIN_SIZE_MM <= s <= MAX_SIZE_MM for s in size):
    reasons.append(f"{where}: each size must be {MIN_SIZE_MM:g} to {MAX_SIZE_MM:g} mm")
    size = None
  if size is not None and shape == "slab" and size[2] * SLAB_RATIO > min(size[0], size[1]):
    reasons.append(f"{where}: a slab is a board, its third size its thickness, at most "
                   f"1/{SLAB_RATIO:g} of the other two; a thicker part is a box")
  pos = _nums(raw.get("pos"), f"{where}: pos", reasons, 3, MAX_POS_MM)
  euler = _nums(raw.get("euler", [0, 0, 0]), f"{where}: euler", reasons, 3, 360.0)
  on = raw.get("on")
  if on is not None and not isinstance(on, str):
    reasons.append(f"{where}: on must be a part's id")
    on = None
  mass = _num(raw.get("mass"), f"{where}: mass", reasons, 0.0, MAX_MASS_KG)
  if mass is not None and mass <= 0.0:
    reasons.append(f"{where}: a part has a mass of its own, more than 0 kg")
    mass = None
  if None in (pid, size, pos, euler, mass) or shape not in SHAPES:
    return None
  return Part(id=pid, shape=shape, size=tuple(s / 1000.0 for s in size),
              pos=tuple(p / 1000.0 for p in pos),
              euler=tuple(math.radians(e) for e in euler), on=on, mass=mass)


def _joint(raw, i: int, reasons: list[str]) -> Joint | None:
  where = f"joint {i}"
  if not isinstance(raw, dict):
    reasons.append(f"{where}: must be an object")
    return None
  jid = _id(raw, where, reasons)
  where = f"joint {jid!r}" if jid else where
  _fields(raw, JOINT_FIELDS, where, reasons)
  kind = raw.get("type")
  if kind not in JOINT_TYPES:
    reasons.append(f"{where}: type must be one of {', '.join(JOINT_TYPES)}")
  part = raw.get("part")
  if not isinstance(part, str):
    reasons.append(f"{where}: part must name the part it moves")
  at = _nums(raw.get("at"), f"{where}: at", reasons, 3, MAX_POS_MM)
  # bounded, or its squares overflow and a direction of 1e200 reads as zero
  axis = _nums(raw.get("axis"), f"{where}: axis", reasons, 3, MAX_AXIS)
  if axis is not None:
    norm = math.sqrt(sum(a * a for a in axis))
    if norm < 1e-9:
      reasons.append(f"{where}: axis is zero")
      axis = None
    else:
      axis = tuple(a / norm for a in axis)
  # a hinge's angles are degrees at the door, a slide's lengths mm
  scale, unit, bound = ((math.pi / 180.0, "deg", MAX_HINGE_DEG) if kind != "slide"
                        else (1e-3, "mm", MAX_SIZE_MM))
  rng = None
  if "range" in raw:
    rng = _nums(raw["range"], f"{where}: range", reasons, 2, bound)
    if rng is not None and not rng[0] < rng[1]:
      reasons.append(f"{where}: range must be [lo, hi] with lo < hi")
      rng = None
    elif rng is not None and not rng[0] <= 0.0 <= rng[1]:
      reasons.append(f"{where}: range [{rng[0]:g}, {rng[1]:g}] {unit} leaves out the pose "
                     f"the document describes, its 0")
      rng = None
    elif rng is not None:
      rng = (rng[0] * scale, rng[1] * scale)
  dyn = {}
  for key in ("stiffness", "damping", "friction"):
    dyn[key] = _num(raw.get(key, 0.0), f"{where}: {key}", reasons, 0.0)
  slack = _num(raw.get("slack", 0.0), f"{where}: slack", reasons, -bound, bound)
  if None in (jid, at, axis, slack, *dyn.values()) or kind not in JOINT_TYPES \
     or not isinstance(part, str) or ("range" in raw and rng is None):
    return None
  return Joint(id=jid, type=kind, part=part, at=tuple(a / 1000.0 for a in at), axis=axis,
               range=rng, slack=slack * scale, **dyn)


def _catch(raw, i: int, reasons: list[str]) -> Catch | None:
  where = f"catch {i}"
  if not isinstance(raw, dict):
    reasons.append(f"{where}: must be an object")
    return None
  _fields(raw, CATCH_FIELDS, where, reasons)
  joint = raw.get("joint")
  if not isinstance(joint, str):
    reasons.append(f"{where}: joint must name the joint it holds")
  at = _nums(raw.get("at"), f"{where}: at", reasons, 3, MAX_POS_MM)
  release = _num(raw.get("release"), f"{where}: release", reasons, 0.0, MAX_RELEASE_N)
  if release is not None and release <= 0.0:
    reasons.append(f"{where}: release is the pull it lets go at, more than 0 N")
    release = None
  if not isinstance(joint, str) or at is None or release is None:
    return None
  return Catch(joint=joint, at=tuple(a / 1000.0 for a in at), release=release)


def _list(raw: dict, key: str, cap: int, reasons: list[str], need: bool) -> list:
  v = raw.get(key, None if need else [])
  if not isinstance(v, list) or (need and not v):
    reasons.append(f"{key} must be a {'non-empty ' if need else ''}list")
    return []
  if len(v) > cap:
    reasons.append(f"{len(v)} {key}; a document may have at most {cap}")
  return v


def parse(raw) -> Scene:
  """A document as its author wrote it -> a `Scene` in the sim's units, or
  `Refused` with every reason at once."""
  if not isinstance(raw, dict):
    raise Refused(["a document is an object of parts, joints and catches"])
  reasons: list[str] = []
  _fields(raw, DOC_FIELDS, "the document", reasons)
  parts = [p for i, r in enumerate(_list(raw, "parts", MAX_PARTS, reasons, True))
           if (p := _part(r, i, reasons)) is not None]
  joints = [j for i, r in enumerate(_list(raw, "joints", MAX_JOINTS, reasons, False))
            if (j := _joint(r, i, reasons)) is not None]
  catches = [c for i, r in enumerate(_list(raw, "catches", MAX_CATCHES, reasons, False))
             if (c := _catch(r, i, reasons)) is not None]
  by_id: dict[str, Part] = {}
  for p in parts:
    if p.id in by_id:
      reasons.append(f"part {p.id!r}: id used twice")
    by_id[p.id] = p
  # what rides what: a tree on the map
  for p in parts:
    if p.on is None:
      continue
    if p.on not in by_id:
      reasons.append(f"part {p.id!r}: on {p.on!r}, which is not a part")
      continue
    seen, cur = {p.id}, by_id[p.on]
    while cur.on is not None and cur.on in by_id:
      if cur.id in seen:
        reasons.append(f"part {p.id!r}: what it is on comes back round to it")
        break
      seen.add(cur.id)
      cur = by_id[cur.on]
  jby: dict[str, Joint] = {}
  moved: set[str] = set()
  for j in joints:
    if j.id in jby:
      reasons.append(f"joint {j.id!r}: id used twice")
    jby[j.id] = j
    if j.part not in by_id:
      reasons.append(f"joint {j.id!r}: moves {j.part!r}, which is not a part")
    elif j.part in moved:
      reasons.append(f"part {j.part!r}: moved by two joints; a part has at most one")
    moved.add(j.part)
  caught: set[str] = set()
  for c in catches:
    j = jby.get(c.joint)
    if j is None:
      reasons.append(f"catch on {c.joint!r}: not a joint")
      continue
    if c.joint in caught:
      reasons.append(f"catch on {c.joint!r}: a joint has at most one catch")
    caught.add(c.joint)
    if j.range is None or (j.range[0] != 0.0 and j.range[1] != 0.0):
      reasons.append(f"catch on {c.joint!r}: a catch holds its joint at an end of its "
                     f"range, where the document's pose is: the range must start or "
                     f"end at 0")
    if j.type == "hinge" and lever(j, c.at) < MIN_LEVER_MM / 1000.0:
      reasons.append(f"catch on {c.joint!r}: its point is on the hinge's axis, so it "
                     f"has no lever")
  if reasons:
    raise Refused(reasons)
  return Scene(parts=tuple(parts), joints=tuple(joints), catches=tuple(catches))


def lever(joint: Joint, point) -> float:
  """A point's distance off a hinge's axis, m (a slide's force has none: 1)."""
  if joint.type == "slide":
    return 1.0
  r = np.asarray(point, dtype=float) - np.asarray(joint.at)
  a = np.asarray(joint.axis)
  return float(np.linalg.norm(r - (r @ a) * a))


# ---- and back -------------------------------------------------------------------

def _mm(vs) -> list[float]:
  return [round(1000.0 * v, 6) for v in vs]


def _deg(vs) -> list[float]:
  return [round(math.degrees(v), 9) for v in vs]


def dump(scene: Scene) -> dict:
  """The document a `Scene` was parsed from, in the author's units: `parse`
  of it is the scene again (the ROUND TRIP)."""
  parts = []
  for p in scene.parts:
    d = {"id": p.id, "shape": p.shape, "size": _mm(p.size), "pos": _mm(p.pos),
         "euler": _deg(p.euler), "mass": p.mass}
    if p.on is not None:
      d["on"] = p.on
    parts.append(d)
  joints = []
  for j in scene.joints:
    conv = _deg if j.type == "hinge" else _mm
    d = {"id": j.id, "type": j.type, "part": j.part, "at": _mm(j.at), "axis": list(j.axis),
         "stiffness": j.stiffness, "slack": conv([j.slack])[0], "damping": j.damping,
         "friction": j.friction}
    if j.range is not None:
      d["range"] = conv(j.range)
    joints.append(d)
  out = {"parts": parts, "joints": joints}
  if scene.catches:
    out["catches"] = [{"joint": c.joint, "at": _mm(c.at), "release": c.release}
                      for c in scene.catches]
  return out
