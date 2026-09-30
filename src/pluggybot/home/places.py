"""What a job may say about where it is sent (issue #419): a building's
rough address and a task area's written directions, and what each task
area's tags are called. World DATA (`places.json`) read here; the rule it
answers to is TaskPattern.md §2's dispatcher test.

THE ADDRESS IS IN THE ROBOT'S MAP. A real robot gets a house's position by
converting its street address once it knows where its own dock is, and its
map is anchored at that dock (#378: the dock is the map's origin); here the
dock is commissioned where the world puts it, so that conversion is the
identity and an address is the building's middle, off by its deliberate
error (`houses`), in map metres.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

PLACES_JSON = Path(__file__).with_name("places.json")


@lru_cache(maxsize=None)
def load(path: str | None = None) -> dict:
  """`places.json`, read once."""
  return json.loads(Path(path or PLACES_JSON).read_text())


def house_middle(house: str) -> tuple[float, float]:
  """The middle of a building's rooms (`home.world.ZONES` naming it), m."""
  from pluggybot.home import world as home
  rooms = [z for z in home.ZONES if z.get("building") == house]
  if not rooms:
    raise ValueError(f"no building {house!r} in the home world")
  x0 = min(z["min"][0] for z in rooms)
  x1 = max(z["max"][0] for z in rooms)
  y0 = min(z["min"][1] for z in rooms)
  y1 = max(z["max"][1] for z in rooms)
  return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def address(house: str) -> dict:
  """A building's address as a job gives it: where its middle is, off by
  its deliberate error, and how far off the address says it may be."""
  data = load()
  ex, ey = data["houses"][house]
  mx, my = house_middle(house)
  return {"house": house, "x": round(mx + ex, 1), "y": round(my + ey, 1),
          "withinM": float(data["withinM"])}


def area(name: str) -> dict | None:
  """A task area's terms for a job -- `address` and `directions` -- or
  None for a target with no area written."""
  spec = load()["areas"].get(name)
  if spec is None:
    return None
  return {"address": address(spec["house"]), "directions": spec["directions"]}


def areas() -> dict[str, dict]:
  """Every task area's terms, by the name a job targets it by."""
  return {name: area(name) for name in load()["areas"]}


def tag_names() -> dict[int, str]:
  """Every task area's tags, by what its directions call them."""
  return {int(t): name for spec in load()["areas"].values()
          for t, name in spec["tags"].items()}
