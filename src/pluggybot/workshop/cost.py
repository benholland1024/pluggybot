"""What a tool costs to fabricate (issue #168, slice D).

Without a cost the agent designs a magic tool for every problem
(Evaluation.md §7, item 9). Three charges, each honest about what it is:

  POINTS    the parts, at the catalog's price -- a purchasable part costs
            what the seller lists, in points, one point per euro. A
            scaffold box costs its filament by mass. What the agent pays is
            the same table the parts page shows a visitor.
  PRINT     sim-seconds the printer takes, by the scaffold mass: a desktop
            FDM printer lays down roughly a gram a minute of PLA at the
            0.2 mm layers a bracket is printed at (`PRINT_S_PER_G`).
  ASSEMBLY  sim-seconds per part to mount it on the frame and wire it
            (`ASSEMBLE_S_PER_PART`), plus the frame itself.

The robot waits the print and assembly time where it stands, drawing what
an idle robot draws, and pays the points before anything is printed -- a
purchase either happens or does not (`Ledger.spend`, no debt). ⚠ These
three constants are DESIGN DECISIONS, not measurements, and say so; the
catalog prices under them are not.
"""

from __future__ import annotations

from pluggybot.workshop.spec import Tool

#: One point per euro. The reward table pays single digits per job and
#: the catalog's parts run from €2.90 (a micro servo) to €84 (the drive
#: motor), so a tool from small parts is an hour's work and one from the
#: body's parts is a week's -- which is the gradient a cost exists for.
POINTS_PER_EUR = 1.0
#: A 1 kg spool of PLA at the German shops the catalog sources from is
#: €15-25 in 2026; the printed mass of a bracket is grams.
FILAMENT_EUR_PER_KG = 20.0
#: ~1 g/min of PLA at 0.2 mm layers on a desktop FDM printer.
PRINT_S_PER_G = 60.0
#: Mount, wire and check one part; the frame (plate, peg, tag) is one more.
ASSEMBLE_S_PER_PART = 120.0


def price(tool: Tool) -> dict:
  """Every charge, itemised, and the totals the lifecycle acts on."""
  parts = []
  eur = 0.0
  print_g = 0.0
  for p in tool.parts:
    if p.part.kind == "scaffold":
      grams = p.mass * 1000.0
      cost = grams / 1000.0 * FILAMENT_EUR_PER_KG
      print_g += grams
      parts.append({"id": p.id, "part": p.part.id, "eur": round(cost, 2),
                    "printedG": round(grams, 1)})
    else:
      cost = float(p.part.priceEur or 0.0)
      parts.append({"id": p.id, "part": p.part.id, "eur": round(cost, 2)})
    eur += cost
  points = int(round(eur * POINTS_PER_EUR))
  print_s = print_g * PRINT_S_PER_G
  assemble_s = (len(tool.parts) + 1) * ASSEMBLE_S_PER_PART
  return {
    "eur": round(eur, 2),
    "points": points,
    "printedG": round(print_g, 1),
    "printS": round(print_s, 1),
    "assembleS": round(assemble_s, 1),
    "waitS": round(print_s + assemble_s, 1),
    "parts": parts,
  }
