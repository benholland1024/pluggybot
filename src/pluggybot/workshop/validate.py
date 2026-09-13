"""docs/ToolPattern.md §2 as code: the coupling envelope a spec must fit
(issue #168). Every number is a `rack/coupling.py` constant, measured on
the built modules; the validator REFUSES, it does not warn, and a spec has
no field with which to renegotiate.

What is checked, and at which pose:

  mass      plate + peg + the face <= MODULE_MASS_CEILING
  moment    g · Σ m·|x| about the peg axis <= LATCH_MOMENT_NM, at the stow
            pose and at every axis's two ends (reach is far dearer than
            mass: 400 g at 150 mm unseats, 800 g at 0 hangs)
  fork      nothing in the volume the fork's prongs and V-plates occupy
            while they hold the peg -- at every pose, because the fork
            holds the tool while it works
  trays     nothing where the rack's V-trays sit, at the stow pose
  bracket   nothing outboard of the plate in the band z -30..-9 mm, at
            the stow pose: a set-down raises the module 31 mm through the
            tray brackets (the pen's carriage at +37 mm jammed there)
  wall      nothing further out the front than WALL_CLEARANCE past the
            plate, at the stow pose
  power     Σ the parts' draw + the module's ESP32 <= PEG_POWER_W; an
            actuator or a sensor whose draw the catalog does not know
            cannot be budgeted and is refused
  bed       every scaffold box fits the print bed in some orientation

Not checked, and said so: the ~1.5-2 N working-force limit (the validator
cannot know what a tool will push against), transit swing clearance (a
sequencing constraint, ToolPattern §2 "in transit"), and the parked
envelope over the chassis -- a tool that hangs lower than the claw's
pendant is checked by the spike rig (slice B), not here.
"""

from __future__ import annotations

from pluggybot.power import MODULE_IDLE_W
from pluggybot.rack.coupling import (
  BRACKET_BAND_Z, FORK_DROP, FORK_Y, LATCH_MOMENT_NM, MODULE_MASS,
  MODULE_MASS_CEILING, PEG_ABOVE_BODY, PEG_POWER_W, TOOL_HALF_X,
  TOOL_HALF_Y, TRAY_Y, V_HALF_LEN, WALL_CLEARANCE,
)
from pluggybot.workshop.spec import Refused, Tool, aabb, parse, poses

G = 9.81

#: The fork's prongs and V-plates while they hold the peg, in the module
#: frame: outboard of the trays at ±FORK_Y, from a little under the V
#: vertex (PEG_ABOVE_BODY - FORK_DROP) to a little over the peg, and the
#: prongs reaching in from the robot's side (+x). 8 mm either side of the
#: plate line for the plates' own thickness and tilt.
FORK_ZONE = (
  (-0.015, 0.070),
  (FORK_Y - 0.008, FORK_Y + 0.008),
  (PEG_ABOVE_BODY - FORK_DROP - 0.010, PEG_ABOVE_BODY + 0.006),
)
#: The rack's V-tray plates, inboard of the fork at ±TRAY_Y, around the peg
#: line, V_HALF_LEN either way in x.
TRAY_ZONE = (
  (-V_HALF_LEN - 0.002, V_HALF_LEN + 0.002),
  (TRAY_Y - 0.008, TRAY_Y + 0.008),
  (PEG_ABOVE_BODY - 0.012, PEG_ABOVE_BODY + 0.010),
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


def validate(tool: Tool) -> list[str]:
  """Every envelope rule the tool breaks, or `[]`."""
  reasons: list[str] = []

  total = MODULE_MASS + tool.mass
  if total > MODULE_MASS_CEILING + 1e-9:
    reasons.append(f"mass: {total * 1000:.0f} g with the plate and peg, over the "
                   f"{MODULE_MASS_CEILING * 1000:.0f} g class")

  worst_moment, worst_at = 0.0, "stow"
  for label, q in _pose_set(tool):
    placed = poses(tool, q)
    moment = sum(p.mass * G * abs(float(placed[p.id][0][0])) for p in tool.parts)
    if moment > worst_moment:
      worst_moment, worst_at = moment, label
    for p in tool.parts:
      lo, hi = aabb(*placed[p.id], p.half)
      if _overlaps(lo, hi, FORK_ZONE):
        reasons.append(f"fork: {p.id!r} sits where the fork's prongs hold the "
                       f"peg ({label})")
  if worst_moment > LATCH_MOMENT_NM + 1e-9:
    reasons.append(f"moment: {worst_moment:.2f} N·m about the peg at {worst_at}, "
                   f"over the latch's {LATCH_MOMENT_NM:.2f}; reach costs more "
                   f"than mass -- bring parts toward x = 0")

  stowed = poses(tool)
  for p in tool.parts:
    lo, hi = aabb(*stowed[p.id], p.half)
    if _overlaps(lo, hi, TRAY_ZONE):
      reasons.append(f"trays: {p.id!r} sits where the rack's V-trays hold the peg")
    band_lo, band_hi = BRACKET_BAND_Z
    if lo[2] < band_hi and hi[2] > band_lo and (hi[1] > TOOL_HALF_Y or lo[1] < -TOOL_HALF_Y):
      reasons.append(f"bracket: {p.id!r} stands outboard of the plate in the band "
                     f"z {band_lo * 1000:.0f}..{band_hi * 1000:.0f} mm, where a "
                     f"set-down meets the tray brackets; keep it under "
                     f"z = {band_lo * 1000:.0f} mm or within |y| <= "
                     f"{TOOL_HALF_Y * 1000:.0f} mm")
    limit = -(TOOL_HALF_X + WALL_CLEARANCE)
    if lo[0] < limit - 1e-9:
      reasons.append(f"wall: {p.id!r} reaches {-lo[0] * 1000:.0f} mm out the front; "
                     f"the wall is {-limit * 1000:.0f} mm from the plate's centre "
                     f"when the tool is racked")

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
  return max(
    sum(p.mass * G * abs(float(poses(tool, q)[p.id][0][0])) for p in tool.parts)
    for _, q in _pose_set(tool))


__all__ = ["Refused", "check", "validate", "worst_moment"]
