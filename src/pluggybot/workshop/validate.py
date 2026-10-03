"""The coupling envelope a spec must fit, as code (issue #168): on legs the
arm's (#407), every number read from `legs/arm.py` -- the tool envelope
#378 measured -- and `legs/rack.py`, the rack it hangs on. The validator
REFUSES, it does not warn, and a spec has no field with which to
renegotiate. ToolPattern.md §2 has the reasons.

What is checked, and at which pose:

  mass      plate + peg + the face <= `arm.TOOL_MAX_KG` (0.40 kg)
  hangs     at the stow pose the centre of mass under the peg's axis
            within `rack.HUNG_TILT_DEG` (2 deg) of plumb: hung on the trays
            nothing holds it level, and a tool hung off plumb is not hung
            (`rack.on_bay`; the rover's scoop hung 5 deg off)
  side      at the stow pose the centre of mass within `rack.HUNG_SIDE_M`
            of the peg's middle: the trays hold the peg at +-45 mm
  ahead     the centre of mass on the peg or ahead of it (away from the
            robot) by at most `arm.TOOL_MAX_AHEAD_M` (60 mm), at the stow
            pose and at every axis's two ends: ahead leans the tool onto
            the fork's lean-pad, behind sits on the pad's post
  moment    g * the mass * that lever <= `arm.TOOL_MAX_MOMENT_NM`
  drop      nothing more than `arm.TOOL_MAX_DROP_M` under the peg, at
            every pose: carried, it must clear the LIDAR's scan plane
  behind    nothing behind the plate's back face, at every pose: that is
            where the fork's prongs, bridge and lean-pad are
  fork      nothing where the fork's V's and end-ramps hold the peg's ends
  trays     nothing where the rack's V-trays and their brackets hold the
            peg, at the stow pose
  board     nothing nearer the rack's back board than `BOARD_CLEAR_M`, at
            the stow pose
  tags      nothing in front of the bay's tags, at the stow pose, from the
            board to the plate's back face: hung, it would hide them from
            the working pose, and no fetch of it would fit its bay (the
            claw's crossbar did, #407)
  rail      nothing in reach of the rail the trays hang from, hung or
            lifted off the trays by a pick (`arm.LIFT`)
  power     the parts' draw + the module's ESP32 <= `coupling.PEG_POWER_W`;
            an actuator or a sensor whose draw the catalog does not know
            cannot be budgeted and is refused
  bed       every scaffold box fits the print bed in some orientation

Not checked, and said so: the force a tool may push with (the validator
cannot know what it will push against). Whether it is taken, works and
hangs back is the rig's, which has gravity and contacts: the workshop runs
it before a point moves (`workshop.build.trial`).
"""

from __future__ import annotations

import math

from pluggybot.legs import arm as am
from pluggybot.legs import rack as rk
from pluggybot.power import MODULE_IDLE_W
from pluggybot.rack.coupling import PEG_ABOVE_BODY, PEG_POWER_W, TOOL_HALF_X
from pluggybot.workshop.spec import Refused, Tool, aabb, parse, poses

G = 9.81

#: The plate and its 220 mm peg, which every built tool has: `legs.rack`'s.
MODULE_MASS = rk.MODULE_MASS
#: The fork's V's and end-ramps round the peg's two ends, in the module
#: frame: from the peg's middle outward past the ramps, a V's flanks either
#: side of the axis, from under its vertex to over the flanks' tops.
_FORK = am.ArmSpec().fork
FORK_ZONE = (
  (-_FORK.v_half_len - 0.004, _FORK.v_half_len + 0.004),
  (_FORK.fork_y - _FORK.v_half_w - 0.010, _FORK.stop_y + _FORK.ramp_w + 0.004),
  (PEG_ABOVE_BODY - _FORK.seat_rise() - 0.016, PEG_ABOVE_BODY + _FORK.flank_top()),
)
#: The rack's V-trays and their brackets, hung (the tool faces the robot, so
#: its frame is the rack's): either side of a bay at `TRAY_Y`, from under a
#: tray's V up to the rail.
TRAY_ZONE = (
  (-0.030, rk.V_HALF_LEN + 0.004),
  (rk.TRAY_Y - rk.TRAY_HALF_W - 0.004, rk.TRAY_Y + rk.TRAY_HALF_W + 0.004),
  (PEG_ABOVE_BODY - rk.TRAY_VERTEX_DROP - 0.016,
   PEG_ABOVE_BODY + rk.DEFAULT.rail_z - rk.DEFAULT.peg_z),
)
#: Nothing nearer the rack's back board than this, hung, m: what its face
#: is from the peg less a margin a hung tool's lean takes.
BOARD_CLEAR_M = 0.010
BOARD_X = rk.DEFAULT.back_x + 0.006
#: The bay's two tags as the working pose's camera sees them past a hung
#: tool: either side of the bay from inside their inner edge's sight line
#: (a part there hides them) out past their outer edge, over their height.
_TAG_HALF = 0.5 * rk.DEFAULT.tag_size * 10 / 8
TAG_ZONE = (
  (BOARD_X, TOOL_HALF_X + 0.002),
  (rk.DEFAULT.tag_dy - _TAG_HALF - 0.012, rk.DEFAULT.tag_dy + _TAG_HALF),
  (PEG_ABOVE_BODY + rk.DEFAULT.tag_z - rk.DEFAULT.peg_z - _TAG_HALF,
   PEG_ABOVE_BODY + rk.DEFAULT.tag_z - rk.DEFAULT.peg_z + _TAG_HALF),
)
#: The rail the trays hang from, across the whole board: out from the board
#: by `rack.RAIL_DEPTH`, and reached from as low as a pick lifts a tool
#: (`arm.LIFT`) under its underside. A thin handle up to 150 mm over the
#: peg jammed on it at 15 deg (#407).
_RAIL_Z = PEG_ABOVE_BODY + rk.DEFAULT.rail_z - rk.DEFAULT.peg_z
RAIL_ZONE = (
  (rk.DEFAULT.back_x - 0.004, rk.DEFAULT.back_x + rk.RAIL_DEPTH + 0.004),
  (-1.0, 1.0),
  (_RAIL_Z - rk.RAIL_HALF_H - am.LIFT - 0.004, _RAIL_Z + rk.RAIL_HALF_H + 0.004),
)


def _overlaps(lo, hi, zone, mirror_y: bool = True) -> bool:
  """Does an AABB touch a zone (and, mirrored, its twin at -y)?"""
  (zx, zy, zz) = zone
  ys = [zy, (-zy[1], -zy[0])] if mirror_y else [zy]
  for y in ys:
    if (lo[0] < zx[1] and hi[0] > zx[0] and lo[1] < y[1] and hi[1] > y[0]
        and lo[2] < zz[1] and hi[2] > zz[0]):
      return True
  return False


def _pose_set(tool: Tool) -> list[tuple[str, dict[str, float]]]:
  """The stow pose, then each axis at each end with the others stowed."""
  out = [("stow", {})]
  for p in tool.axes:
    a = p.axis
    assert a is not None
    out.append((f"{a.verb} at its low end", {a.verb: a.lo}))
    out.append((f"{a.verb} at its high end", {a.verb: a.hi}))
  return out


def ahead_of_peg(tool: Tool, q: dict[str, float] | None = None) -> float:
  """How far the tool's centre of mass sits AHEAD of its peg (away from
  the robot: the module's -x), m, the plate and peg on the peg's line."""
  placed = poses(tool, q)
  moment = sum(p.mass * float(placed[p.id][0][0]) for p in tool.parts)
  return -moment / (MODULE_MASS + tool.mass)


def side_of_peg(tool: Tool) -> float:
  """How far the tool's centre of mass sits along its peg from the peg's
  middle at the stow pose, m, signed (+y), the plate and peg centred."""
  placed = poses(tool)
  return sum(p.mass * float(placed[p.id][0][1]) for p in tool.parts) / (MODULE_MASS + tool.mass)


def hang_tilt_deg(tool: Tool) -> float:
  """How far off plumb the tool hangs from its peg on the trays, deg: its
  centre of mass at the stow pose swung under the peg's axis (the plate's
  at its middle, the peg's on its axis). 90 for one at or over the axis."""
  placed = poses(tool)
  total = MODULE_MASS + tool.mass
  peg = rk.peg_kg()
  x = sum(p.mass * float(placed[p.id][0][0]) for p in tool.parts) / total
  z = (sum(p.mass * float(placed[p.id][0][2]) for p in tool.parts)
       + peg * PEG_ABOVE_BODY) / total
  depth = PEG_ABOVE_BODY - z
  return 90.0 if depth <= 0.0 else math.degrees(math.atan2(abs(x), depth))


def validate(tool: Tool) -> list[str]:
  """Every envelope rule the tool breaks, or `[]`."""
  reasons: list[str] = []

  total = MODULE_MASS + tool.mass
  tilt = hang_tilt_deg(tool)
  if tilt > rk.HUNG_TILT_DEG + 1e-9:
    reasons.append(f"hangs: hung on the trays it would tilt {tilt:.1f} deg off plumb, "
                   f"past the {rk.HUNG_TILT_DEG:g} a hung tool may; at its stow pose bring "
                   "its centre of mass under the peg")
  side = side_of_peg(tool)
  if abs(side) > rk.HUNG_SIDE_M + 1e-9:
    reasons.append(f"side: at its stow pose its centre of mass sits {abs(side) * 1000:.0f} mm "
                   f"to one side of the peg's middle, past the {rk.HUNG_SIDE_M * 1000:.0f} a "
                   f"hung tool may; the trays hold the peg {rk.TRAY_Y * 1000:.0f} mm either side")
  if total > am.TOOL_MAX_KG + 1e-9:
    reasons.append(f"mass: {total * 1000:.0f} g with the plate and peg, over the "
                   f"arm's {am.TOOL_MAX_KG * 1000:.0f} g")

  worst, worst_at = 0.0, "stow"
  behind, deepest = [], []
  for label, q in _pose_set(tool):
    ahead = ahead_of_peg(tool, q)
    if abs(ahead) > abs(worst):
      worst, worst_at = ahead, label
    placed = poses(tool, q)
    for p in tool.parts:
      lo, hi = aabb(*placed[p.id], p.half)
      if hi[0] > TOOL_HALF_X + 0.002 and p.id not in behind:
        behind.append(p.id)
        reasons.append(f"behind: {p.id!r} stands behind the plate ({label}), where the "
                       "fork's prongs and its lean-pad are; build ahead of the plate")
      if lo[2] < PEG_ABOVE_BODY - am.TOOL_MAX_DROP_M - 1e-9 and p.id not in deepest:
        deepest.append(p.id)
        reasons.append(f"drop: {p.id!r} hangs {(PEG_ABOVE_BODY - lo[2]) * 1000:.0f} mm under "
                       f"the peg ({label}), past the {am.TOOL_MAX_DROP_M * 1000:.0f} mm a "
                       "carried tool may: it would cross the LIDAR's scan plane")
      if _overlaps(lo, hi, FORK_ZONE):
        reasons.append(f"fork: {p.id!r} sits where the fork's V's hold the peg ({label})")
  if worst < -0.005:
    reasons.append(f"ahead: the centre of mass sits {-worst * 1000:.0f} mm BEHIND the peg "
                   f"at {worst_at}, on the lean-pad's post; keep it on the peg's line or ahead")
  elif worst > am.TOOL_MAX_AHEAD_M + 1e-9:
    reasons.append(f"ahead: the centre of mass sits {worst * 1000:.0f} mm ahead of the peg "
                   f"at {worst_at}, past the arm's {am.TOOL_MAX_AHEAD_M * 1000:.0f} mm; "
                   "bring parts toward the peg")
  moment = G * total * max(worst, 0.0)
  if moment > am.TOOL_MAX_MOMENT_NM + 1e-9:
    reasons.append(f"moment: {moment:.2f} N·m about the peg at {worst_at}, over the "
                   f"arm's {am.TOOL_MAX_MOMENT_NM:.2f}")

  stowed = poses(tool)
  for p in tool.parts:
    lo, hi = aabb(*stowed[p.id], p.half)
    if _overlaps(lo, hi, TRAY_ZONE):
      reasons.append(f"trays: {p.id!r} sits where the rack's V-trays and their "
                     "brackets hold the peg")
    if lo[0] < BOARD_X + BOARD_CLEAR_M - 1e-9:
      reasons.append(f"board: {p.id!r} reaches {-lo[0] * 1000:.0f} mm ahead of the peg; "
                     f"hung, the rack's back board is {-BOARD_X * 1000:.0f} mm off it")
    if _overlaps(lo, hi, TAG_ZONE):
      reasons.append(f"tags: {p.id!r} stands in front of its bay's tags, hung; the robot "
                     "would not see them from the working pose, and no fetch of it "
                     "would fit its bay")
    if _overlaps(lo, hi, RAIL_ZONE, mirror_y=False):
      reasons.append(f"rail: {p.id!r} reaches the rail the trays hang from, "
                     f"{(_RAIL_Z - rk.RAIL_HALF_H - PEG_ABOVE_BODY) * 1000:.0f} mm over the peg, "
                     f"hung or lifted {am.LIFT * 1000:.0f} mm off the trays by a pick")

  draw = MODULE_IDLE_W
  for p in tool.parts:
    w = p.part.capabilities.get("powerW")
    if p.part.kind in ("actuator", "sensor", "electronics"):
      if w is None:
        reasons.append(f"power: the catalog does not know what {p.part.id} draws "
                       f"({p.id!r}), so the peg's budget cannot be checked")
      else:
        draw += float(w)
  if draw > PEG_POWER_W + 1e-9:
    reasons.append(f"power: {draw:.1f} W through the peg with the module's own "
                   f"{MODULE_IDLE_W:g} W, over its {PEG_POWER_W:g} W")

  for p in tool.parts:
    if p.part.kind != "scaffold":
      continue
    bed = p.part.capabilities["printBedMm"]
    bed_sorted = sorted(float(bed[k]) for k in ("x", "y", "z"))
    size_sorted = sorted(h * 2000.0 for h in p.half)
    if any(s > b + 1e-9 for s, b in zip(size_sorted, bed_sorted)):
      reasons.append(f"bed: {p.id!r} is {' × '.join(f'{s:g}' for s in size_sorted)} mm, "
                     f"more than the {' × '.join(f'{b:g}' for b in bed_sorted)} mm "
                     f"print bed in any orientation")
  return reasons


def check(raw) -> Tool:
  """Parse and validate in one door: a `Tool`, or `Refused` with every
  structural and envelope reason together."""
  tool = parse(raw)
  reasons = validate(tool)
  if reasons:
    raise Refused(reasons)
  return tool


def worst_moment(tool: Tool) -> float:
  """For a report: the largest moment about the peg over the pose set."""
  total = MODULE_MASS + tool.mass
  return max(G * total * max(ahead_of_peg(tool, q), 0.0) for _, q in _pose_set(tool))


__all__ = ["Refused", "check", "side_of_peg", "validate", "worst_moment"]
