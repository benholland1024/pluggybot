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

A TEMPLATE (`parse_template`) is a document some of whose numbers are not
known yet, each written {"between": [lo, hi]} in its field's units, for a
fitter to find (`fit.py`): what a part weighs, and how a joint or a catch
holds (`UNKNOWABLE`). ⚠ THE GEOMETRY IS MEASURED, NEVER FITTED (#466): a
size, a place, an axis or a range written so is refused. `placed` writes a
document drawn in a frame of its own -- a box's, square to it -- into the
map's.

Nothing here is specific to a lid.
"""

from __future__ import annotations

import copy
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
#: What a template may leave unknown, by what it is in: what the robot
#: cannot see. Everything else is measured.
UNKNOWABLE = {"part": ("mass",), "joint": ("stiffness", "slack", "damping", "friction"),
              "catch": ("release",)}
#: A template's unknowns at most: each costs the fitter a rollout a step.
MAX_UNKNOWNS = 12


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


def euler_of(rot) -> tuple[float, float, float]:
  """`rotation`'s angles back from a rotation, radians (the middle one
  within +-90 deg)."""
  r = np.asarray(rot, dtype=float)
  b = math.asin(max(-1.0, min(1.0, float(r[0, 2]))))
  return (math.atan2(-float(r[1, 2]), float(r[2, 2])), b,
          math.atan2(-float(r[0, 1]), float(r[0, 0])))


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


def _unknown(v) -> bool:
  """A template's unknown, as written: {"between": [lo, hi]}."""
  return isinstance(v, dict) and "between" in v


def _num(v, name: str, reasons: list[str], lo: float | None = None,
         hi: float | None = None) -> float | None:
  if _unknown(v):
    reasons.append(f"{name} is unknown: a document to compile has every number "
                   f"(a template's are filled by the fitter)")
    return None
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


# ---- templates: what the robot cannot see, left for the fitter ------------------------

#: Each list a template's unknowns can be in, and what its items are.
_KINDS = (("parts", "part"), ("joints", "joint"), ("catches", "catch"))


@dataclass(frozen=True)
class Unknown:
  """One number a template leaves to the fitter: `name` (`<id>.<field>`, a
  catch's `<joint>.release`), where it is written (its list, its index
  there, its field) and its range, in the author's units."""
  name: str
  kind: str
  index: int
  field: str
  lo: float
  hi: float

  def at(self, u: float) -> float:
    """The value a share `u` (0 to 1) of the way through its range."""
    return self.lo + float(u) * (self.hi - self.lo)

  def share(self, value: float) -> float:
    return (float(value) - self.lo) / (self.hi - self.lo)


@dataclass(frozen=True)
class Template:
  """A document as written, unknowns and all (`raw`), and its `unknowns` in
  the order they were written."""
  raw: dict
  unknowns: tuple[Unknown, ...] = ()

  @property
  def names(self) -> list[str]:
    return [u.name for u in self.unknowns]

  def fill(self, values) -> dict:
    """The document with each unknown at its value (author's units), in
    `unknowns`' order."""
    values = list(values)
    if len(values) != len(self.unknowns):
      raise ValueError(f"{len(values)} values for {len(self.unknowns)} unknowns")
    doc = copy.deepcopy(self.raw)
    for u, v in zip(self.unknowns, values):
      doc[u.kind][u.index][u.field] = float(v)
    return doc

  def middle(self) -> list[float]:
    return [u.at(0.5) for u in self.unknowns]


def _item_name(kind: str, item: dict) -> str:
  key = item.get("joint" if kind == "catch" else "id")
  return key if isinstance(key, str) else "?"


def parse_template(raw) -> Template:
  """A template as its author wrote it -> a `Template`, or `Refused` with
  every reason at once: an unknown's range, a geometry field written as
  one, and everything `parse` refuses of the document filled at either end
  of every range."""
  if not isinstance(raw, dict):
    raise Refused(["a template is an object of parts, joints and catches"])
  reasons: list[str] = []
  unknowns: list[Unknown] = []
  for key, kind in _KINDS:
    items = raw.get(key, [])
    if not isinstance(items, list):
      continue                                          # `parse` says why
    for i, item in enumerate(items):
      if not isinstance(item, dict):
        continue
      name = _item_name(kind, item)
      where = f"catch on {name!r}" if kind == "catch" else f"{kind} {name!r}"
      for field, v in item.items():
        if isinstance(v, list) and any(_unknown(x) for x in v):
          reasons.append(f"{where}: {field} is measured, never fitted, so it is written "
                         f"as it was measured")
          continue
        if not _unknown(v):
          continue
        if field not in UNKNOWABLE[kind]:
          reasons.append(f"{where}: {field} is measured, never fitted, so it is written as "
                         f"it was measured; what may be unknown on a {kind} is "
                         f"{' or '.join(UNKNOWABLE[kind])}")
          continue
        b = v["between"]
        if set(v) != {"between"} or not isinstance(b, list) or len(b) != 2 \
           or not all(_finite(x) for x in b):
          reasons.append(f"{where}: an unknown {field} is {{\"between\": [lo, hi]}}, two "
                         f"numbers and nothing else")
          continue
        lo, hi = float(b[0]), float(b[1])
        if not lo < hi:
          reasons.append(f"{where}: {field} between [{lo:g}, {hi:g}] must be [lo, hi] with "
                         f"lo < hi (a number that is known is written as a number)")
          continue
        unknowns.append(Unknown(name=f"{name}.{field}", kind=key, index=i, field=field,
                                lo=lo, hi=hi))
  if len(unknowns) > MAX_UNKNOWNS:
    reasons.append(f"{len(unknowns)} unknowns; a template may have at most {MAX_UNKNOWNS}")
  names = [u.name for u in unknowns]
  for n in sorted({n for n in names if names.count(n) > 1}):
    reasons.append(f"{n} is unknown twice")
  template = Template(raw=copy.deepcopy(raw), unknowns=tuple(unknowns))
  # the document at both ends of every range: its structure, and each end
  # against its field's own bounds
  for end in ("lo", "hi"):
    try:
      parse(template.fill([getattr(u, end) for u in unknowns]))
    except Refused as e:
      reasons += [r for r in e.reasons if r not in reasons]
  if reasons:
    raise Refused(reasons)
  return template


# ---- a document drawn in a frame of its own ----------------------------------------

def _xyz(v) -> bool:
  return isinstance(v, list) and len(v) == 3 and all(_finite(x) for x in v)


def placed(raw: dict, at, yaw_deg: float) -> dict:
  """A document (or a template) drawn in a frame of its own -- its origin
  at `at` (x, y, z mm in the map), turned `yaw_deg` about the map's z --
  written in the map's: every part's place and turn, every joint's point
  and axis and every catch's point. Unknowns are dynamics, and stay as they
  were; a field that is not numbers is left for `parse` to refuse."""
  doc = copy.deepcopy(raw)
  y = math.radians(float(yaw_deg))
  c, s = math.cos(y), math.sin(y)
  turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
  origin = np.asarray(at, dtype=float)

  def point(p) -> list[float]:
    return [float(v) for v in turn @ np.asarray(p, dtype=float) + origin]

  for part in doc.get("parts", []) if isinstance(doc.get("parts"), list) else []:
    if not isinstance(part, dict):
      continue
    if _xyz(part.get("pos")):
      part["pos"] = point(part["pos"])
    euler = part.get("euler", [0.0, 0.0, 0.0])
    if _xyz(euler):
      rot = turn @ rotation([math.radians(e) for e in euler])
      part["euler"] = [math.degrees(e) for e in euler_of(rot)]
  for key, fields in (("joints", ("at", "axis")), ("catches", ("at",))):
    for item in doc.get(key, []) if isinstance(doc.get(key), list) else []:
      if not isinstance(item, dict):
        continue
      for f in fields:
        if _xyz(item.get(f)):
          item[f] = (point(item[f]) if f == "at"
                     else [float(v) for v in turn @ np.asarray(item[f], dtype=float)])
  return doc
