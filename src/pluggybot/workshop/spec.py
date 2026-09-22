"""The tool spec: what an agent emits to describe a tool (issue #168).

JSON, and small on purpose. The plate, the peg and the identity tag are
NOT in it -- `module_xml` generates them, and a tool that could move its own
peg could leave the envelope. What the agent says is which catalog parts
go where, in the module's own frame (+x toward the robot, -x at the wall
when stowed, z up, the peg axis along y at z = `PEG_ABOVE_BODY`):

  {"name": "scoop",
   "parts": [
     {"id": "hinge", "part": "servo_fs90", "pos": [-20, 0, -40],
      "axis": {"verb": "tilt", "dir": [0, 1, 0], "range": [0, 90], "stow": 0}},
     {"id": "blade", "part": "scaffold_pla_box", "size": [60, 30, 4],
      "pos": [-30, 0, -8], "on": "hinge"}]}

Per part: `id`, `part` (a catalog id on the `catalog` shelf), `pos` in mm
in its parent's frame, optional `euler` in degrees, optional `on` (another
part's id; default the frame -- the assembly graph is a tree rooted at the
plate). A `scaffold` part carries its `size` in mm; every other part's
extent is the catalog's `dimensionsMm`. An `actuator` part carries an
`axis`: the `verb` a program will `move()`, its `dir`, its `range` (degrees
for a hinge, mm for a slide -- the part's `motion` says which) and its
`stow` position; a part `on` an actuator rides that axis.

  ⚠ UNITS ARE THE AGENT'S (mm, degrees) AT THE DOOR AND THE SIM'S (m,
  radians) INSIDE. `parse` converts once; nothing after it sees a millimetre.

  ⚠ AN UNKNOWN FIELD IS REFUSED, NOT IGNORED. There is no field for a solver
  option, a contact parameter, a mass or a friction -- those come from the
  catalog and the generator -- and a spec that could carry one would be a
  spec that could set it. `test_workshop.py` pins this with `noslip`.

`parse` is STRUCTURE: does the spec name real parts and hang together. The
envelope -- mass, moment, clearance, stow, power, print bed -- is
`validate.py`. Both return every reason at once, on `procedure/steps.py`'s
terms, so an author fixes them together.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from pluggybot.rack import catalog
from pluggybot.rack.tags import MODULE_TAG_IDS

#: How many parts a tool may have. The claw is ~6 geoms, the dispenser ~10;
#: a tool past this is a rack, not a module.
MAX_PARTS = 12
NAME = re.compile(r"^[a-z][a-z0-9_]{0,15}$")
ID = re.compile(r"^[a-z][a-z0-9_]{0,15}$")
FRAME = "frame"

PART_FIELDS = {"id", "part", "pos", "euler", "on", "size", "axis"}
AXIS_FIELDS = {"verb", "dir", "range", "stow"}
SPEC_FIELDS = {"name", "parts"}


class Refused(ValueError):
  """A spec that does not parse or validate. `reasons` is every rule it
  broke, so an author fixes them all at once rather than one per refusal."""

  def __init__(self, reasons: list[str]) -> None:
    super().__init__("; ".join(reasons))
    self.reasons = list(reasons)


@dataclass(frozen=True)
class Axis:
  """One actuated degree of freedom a tool brings, in sim units."""
  verb: str
  kind: str                 # "hinge" | "slide", from the part's `motion`
  direction: tuple[float, float, float]
  lo: float                 # rad or m
  hi: float
  stow: float
  speed: float              # rad/s or m/s, the part's own ceiling
  force: float              # N·m or N, the part's own limit


@dataclass(frozen=True)
class Placed:
  """One catalog part at a pose in its parent's frame."""
  id: str
  part: catalog.Part
  pos: tuple[float, float, float]        # m, in the parent's frame
  euler: tuple[float, float, float]      # rad
  on: str                                # FRAME or a part id
  shape: str                             # "box" | "cylinder"
  half: tuple[float, float, float]       # m: box half-extents, or (r, r, half-length)
  mass: float                            # kg
  axis: Axis | None = None

  @property
  def actuated(self) -> bool:
    return self.axis is not None


@dataclass
class Tool:
  name: str
  parts: tuple[Placed, ...]
  by_id: dict[str, Placed] = field(init=False)

  def __post_init__(self) -> None:
    self.by_id = {p.id: p for p in self.parts}

  @property
  def body(self) -> str:
    return f"module_{self.name}"

  @property
  def axes(self) -> list[Placed]:
    return [p for p in self.parts if p.axis is not None]

  def children(self, of: str) -> list[Placed]:
    return [p for p in self.parts if p.on == of]

  @property
  def mass(self) -> float:
    """The face's mass; the plate and peg are `coupling.MODULE_MASS` on top."""
    return sum(p.mass for p in self.parts)


# ---- geometry helpers ------------------------------------------------------

def rotation(euler_rad: tuple[float, float, float]) -> np.ndarray:
  """MuJoCo's default euler sequence: intrinsic x-y-z, radians."""
  x, y, z = euler_rad
  cx, sx, cy, sy, cz, sz = map(float, (math.cos(x), math.sin(x), math.cos(y),
                                        math.sin(y), math.cos(z), math.sin(z)))
  rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
  ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
  rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
  return rx @ ry @ rz


def axis_rotation(direction, angle: float) -> np.ndarray:
  """Rodrigues: rotation of `angle` radians about a unit `direction`."""
  k = np.asarray(direction, dtype=float)
  kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
  return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * (kx @ kx)


def poses(tool: Tool, q: dict[str, float] | None = None
          ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
  """Every part's (centre, rotation) in the MODULE frame, with each axis
  at `q[verb]` (default: its stow). A part on an actuator inherits the
  actuator's pose plus the joint's displacement about the actuator's
  own centre."""
  q = q or {}
  out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

  def place(p: Placed) -> tuple[np.ndarray, np.ndarray]:
    if p.id in out:
      return out[p.id]
    if p.on == FRAME:
      base_c, base_r = np.zeros(3), np.eye(3)
    else:
      parent = tool.by_id[p.on]
      base_c, base_r = place(parent)
      if parent.axis is not None:
        a = parent.axis
        value = q.get(a.verb, a.stow)
        d = base_r @ np.asarray(a.direction, dtype=float)
        if a.kind == "slide":
          base_c = base_c + d * value
        else:
          base_r = axis_rotation(d, value) @ base_r
    c = base_c + base_r @ np.asarray(p.pos, dtype=float)
    r = base_r @ rotation(p.euler)
    out[p.id] = (c, r)
    return out[p.id]

  for p in tool.parts:
    place(p)
  return out


def aabb(centre: np.ndarray, rot: np.ndarray, half) -> tuple[np.ndarray, np.ndarray]:
  """A rotated box's axis-aligned bounds: centre ± |R|·half."""
  h = np.abs(rot) @ np.asarray(half, dtype=float)
  return centre - h, centre + h


# ---- parsing ---------------------------------------------------------------

def _num(v, name: str, reasons: list[str], n: int | None = None):
  if n is None:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
      reasons.append(f"{name} must be a number")
      return None
    return float(v)
  if not isinstance(v, list) or len(v) != n or any(
      not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x)
      for x in v):
    reasons.append(f"{name} must be {n} numbers")
    return None
  return tuple(float(x) for x in v)


def extent(part: catalog.Part, size_mm=None) -> tuple[str, tuple[float, float, float]] | str:
  """A part's primitive and half-extents in metres, or WHY it has none.

  A scaffold box is whatever the spec asks for. A catalog part is the box
  or cylinder its `dimensionsMm` describe -- and a part whose dimensions
  the catalog does not know cannot be placed, because a part with no
  extent cannot be checked for clearance."""
  if part.kind == "scaffold":
    if size_mm is None:
      return "a scaffold part needs a size"
    return "box", tuple(s / 2000.0 for s in size_mm)
  d = part.dimensionsMm
  if not d:
    return f"the catalog does not know its size ({part.why.get('dimensionsMm', 'no reason')})"
  if {"length", "width", "height"} <= set(d):
    return "box", (d["length"] / 2000.0, d["width"] / 2000.0, d["height"] / 2000.0)
  if {"x", "y", "z"} <= set(d):
    return "box", (d["x"] / 2000.0, d["y"] / 2000.0, d["z"] / 2000.0)
  if {"diameter", "length"} <= set(d):
    r = d["diameter"] / 2000.0
    return "cylinder", (r, r, d["length"] / 2000.0)
  return f"the catalog's dimensions ({', '.join(d)}) do not describe a box or a cylinder"


def unbuildable(part: catalog.Part) -> str | None:
  """Why a catalog part cannot be built from, or None. ONE predicate for
  the validator's refusals and the prompt's list (issue #168 slice D), so
  the robot is never told a part is usable that `parse` would refuse."""
  if "catalog" not in part.shelves:
    return "on the body shelf, not the catalog"
  if part.kind == "scaffold":
    return None
  if part.status != "chosen":
    return part.why.get("partNumber") or f"{part.status}, not chosen"
  if part.massG is None:
    return f"mass unknown ({part.why.get('massG', 'no reason')})"
  ext = extent(part)
  if isinstance(ext, str):
    return ext
  cap = part.capabilities
  if part.kind == "actuator":
    kind = cap.get("motion")
    if kind == "hinge":
      needed = ("angleDeg", "speedDegS", "torqueNm")
    elif kind == "slide":
      needed = ("strokeMm", "speedMmS", "forceN")
    else:
      return "motion unknown"
    for key in needed:
      if cap.get(key) is None:
        return f"{key} unknown"
  if part.kind in ("actuator", "sensor", "electronics") and cap.get("powerW") is None:
    return "draw unknown, so the peg's power budget cannot be checked"
  return None


def buildable() -> list[str]:
  """Every catalog id a tool may actually be built from, in catalog order.
  `unbuildable`'s complement, so the refusal a robot reads and the list the
  prompt carries come off one predicate (issue #315: 433 specs named no
  part at all, and "no catalog part 'blade'" never said what would have
  worked)."""
  return [part.id for part in catalog.PARTS if unbuildable(part) is None]


def _scaffold_mass(part: catalog.Part, half) -> float:
  density = float(part.capabilities["densityKgM3"])
  return density * 8.0 * half[0] * half[1] * half[2]


def _axis(raw, part: catalog.Part, where: str, reasons: list[str]) -> Axis | None:
  if not isinstance(raw, dict):
    reasons.append(f"{where}: axis must be an object")
    return None
  for key in raw:
    if key not in AXIS_FIELDS:
      reasons.append(f"{where}: unknown axis field {key!r}")
  for key in ("verb", "dir", "range", "stow"):
    if key not in raw:
      reasons.append(f"{where}: axis needs {key}")
  if reasons:
    return None
  cap = part.capabilities
  kind = cap.get("motion")
  if kind not in ("hinge", "slide"):
    reasons.append(f"{where}: {part.id} is not an actuator the catalog knows the motion of")
    return None
  verb = raw["verb"]
  if not isinstance(verb, str) or not ID.match(verb):
    reasons.append(f"{where}: axis verb must match {ID.pattern}")
  d = _num(raw["dir"], f"{where}: axis dir", reasons, 3)
  rng = _num(raw["range"], f"{where}: axis range", reasons, 2)
  stow = _num(raw["stow"], f"{where}: axis stow", reasons)
  if d is None or rng is None or stow is None:
    return None
  norm = math.sqrt(sum(x * x for x in d))
  if norm < 1e-9:
    reasons.append(f"{where}: axis dir is zero")
    return None
  d = tuple(x / norm for x in d)
  lo, hi = rng
  if not lo < hi:
    reasons.append(f"{where}: axis range must be [lo, hi] with lo < hi")
    return None
  if not lo <= stow <= hi:
    reasons.append(f"{where}: stow {stow:g} is outside the range [{lo:g}, {hi:g}]")
  # the part's own limits: a hinge in degrees, a slide in mm
  if kind == "hinge":
    limit, speed, force = cap.get("angleDeg"), cap.get("speedDegS"), cap.get("torqueNm")
    unit, scale = "°", math.pi / 180.0
  else:
    limit, speed, force = cap.get("strokeMm"), cap.get("speedMmS"), cap.get("forceN")
    unit, scale = " mm", 1e-3
  for what, value in (("travel", limit), ("speed", speed), ("force", force)):
    if value is None:
      reasons.append(f"{where}: the catalog does not know {part.id}'s {what}, "
                     f"so a tool cannot be built from it")
  if reasons:
    return None
  if hi - lo > limit + 1e-9:
    reasons.append(f"{where}: range spans {hi - lo:g}{unit}, more than the "
                   f"part's {limit:g}{unit}")
    return None
  return Axis(verb=verb, kind=kind, direction=d, lo=lo * scale, hi=hi * scale,
              stow=stow * scale, speed=float(speed) * scale, force=float(force))


def parse(raw) -> Tool:
  """A spec as the agent wrote it -> a `Tool`, or `Refused` with every
  structural reason. Envelope rules are `validate.validate`."""
  reasons: list[str] = []
  if not isinstance(raw, dict):
    raise Refused(["a spec is an object"])
  for key in raw:
    if key not in SPEC_FIELDS:
      reasons.append(f"unknown field {key!r} (a spec names parts and where they go; "
                     f"nothing else)")
  name = raw.get("name")
  if not isinstance(name, str) or not NAME.match(name):
    reasons.append(f"name must match {NAME.pattern}")
    name = "invalid"
  elif f"module_{name}" in MODULE_TAG_IDS:
    reasons.append(f"the rack already has a module_{name}")
  parts_raw = raw.get("parts")
  if not isinstance(parts_raw, list) or not parts_raw:
    # ⚠ THE PART LIST IS WHAT A SPEC IS, and a refusal that only said so
    # in the abstract was what 433 deployed specs got back (issue #315).
    # Reached from a decision only by a hand-written or restored spec now
    # -- `overseer.idle_build` drops a partless one before it arrives.
    raise Refused(reasons + [
      "parts must be a non-empty list: a spec IS its parts, each one "
      "{\"id\", \"part\", \"pos\"} -- a tool may be built from "
      + ", ".join(buildable())])
  if len(parts_raw) > MAX_PARTS:
    reasons.append(f"{len(parts_raw)} parts; the most a tool may have is {MAX_PARTS}")

  parts_by_id = catalog.by_id()
  placed: list[Placed] = []
  ids: set[str] = set()
  for i, pr in enumerate(parts_raw):
    where = f"part {i}"
    if not isinstance(pr, dict):
      reasons.append(f"{where}: must be an object")
      continue
    pid = pr.get("id")
    if isinstance(pid, str) and ID.match(pid):
      where = f"part {pid!r}"
      if pid in ids or pid == FRAME:
        reasons.append(f"{where}: id used twice")
      ids.add(pid)
    else:
      reasons.append(f"{where}: id must match {ID.pattern}")
      pid = f"_{i}"
    for key in pr:
      if key not in PART_FIELDS:
        reasons.append(f"{where}: unknown field {key!r}")
    part = parts_by_id.get(pr.get("part"))
    if part is None:
      reasons.append(f"{where}: no catalog part {pr.get('part')!r} -- a tool may "
                     "be built from " + ", ".join(buildable()))
      continue
    if "catalog" not in part.shelves:
      reasons.append(f"{where}: {part.id} is on the body shelf, not the catalog")
    if part.status != "chosen":
      reasons.append(f"{where}: {part.id} is {part.status}, not chosen"
                     + (f" ({part.why.get('partNumber')})" if part.why.get("partNumber") else ""))
    pos = _num(pr.get("pos"), f"{where}: pos", reasons, 3)
    euler = _num(pr.get("euler", [0, 0, 0]), f"{where}: euler", reasons, 3)
    on = pr.get("on", FRAME)
    if not isinstance(on, str):
      reasons.append(f"{where}: on must be a part id")
      on = FRAME
    size = None
    if "size" in pr:
      if part.kind != "scaffold":
        reasons.append(f"{where}: only a scaffold part takes a size; {part.id}'s "
                       f"is the catalog's")
      else:
        size = _num(pr["size"], f"{where}: size", reasons, 3)
        if size is not None and min(size) <= 0:
          reasons.append(f"{where}: size must be positive")
          size = None
    ext = extent(part, size)
    if isinstance(ext, str):
      reasons.append(f"{where}: {ext}")
      continue
    shape, half = ext
    if part.kind == "scaffold":
      mass = _scaffold_mass(part, half)
    elif part.massG is None:
      reasons.append(f"{where}: the catalog does not know {part.id}'s mass "
                     f"({part.why.get('massG', 'no reason')}), so a tool cannot "
                     f"be built from it")
      continue
    else:
      mass = part.massG / 1000.0
    axis = None
    if "axis" in pr:
      if part.kind != "actuator":
        reasons.append(f"{where}: {part.id} is a {part.kind}, not an actuator; it "
                       f"has no axis")
      else:
        sub: list[str] = []
        axis = _axis(pr["axis"], part, where, sub)
        reasons.extend(sub)
    elif part.kind == "actuator":
      reasons.append(f"{where}: an actuator needs an axis (verb, dir, range, stow)")
    if pos is None or euler is None:
      continue
    placed.append(Placed(id=pid, part=part, pos=tuple(x / 1000.0 for x in pos),
                         euler=tuple(math.radians(e) for e in euler), on=on,
                         shape=shape, half=half, mass=mass, axis=axis))

  # the assembly graph: a tree on the frame, every `on` a real part
  by_id = {p.id: p for p in placed}
  for p in placed:
    if p.on != FRAME and p.on not in by_id:
      reasons.append(f"part {p.id!r}: mounted on {p.on!r}, which is not a part")
  for p in placed:
    seen, cur = set(), p
    while cur.on != FRAME and cur.on in by_id:
      if cur.id in seen:
        reasons.append(f"part {p.id!r}: mounts form a cycle")
        break
      seen.add(cur.id)
      cur = by_id[cur.on]
  verbs = [p.axis.verb for p in placed if p.axis]
  for v in set(verbs):
    if verbs.count(v) > 1:
      reasons.append(f"axis verb {v!r} used twice")
  if reasons:
    raise Refused(reasons)
  return Tool(name=name, parts=tuple(placed))
