"""A* shortest path over a boolean traversable grid (4-connected).

A* is Dijkstra's cheapest-first expansion plus a sense of direction: nodes
are ranked by g + h, where g is the cost already paid from the start and h
is an optimistic estimate of the cost remaining (here Manhattan distance,
which never overestimates on a 4-connected grid — that "admissibility" is
what preserves optimality).
"""

import heapq


def nearest_traversable(traversable, start: tuple[int, int],
                        radius: int = 10) -> tuple[int, int] | None:
  """`start` if it is traversable, else the nearest traversable cell within
  `radius` cells -- or None if there is none, which means the robot is not in
  a halo but off its own map.

  THE ESCAPE FROM THE INFLATION HALO (issue #92). `astar` exempts the start
  cell -- the robot is physically standing on it, whatever the inflated map
  claims -- but the exemption is one cell deep: a robot parked 30 cm from a
  couch has its start cell AND all four neighbours inside the couch's 35 cm
  inflation ring, and cannot take a first step. Measured on the expanded
  home world: exploration gave up at 61 s with "no-reachable" while 366 of
  367 frontiers were reachable, two thirds of the house unmapped, and the
  nearest traversable cell 15 cm away.

  This existed in `HubMission._plan_to` (so `drive_to` never trapped) and
  not in `navigation.plan` (so exploration did) -- two implementations of
  the same idea, one of which had quietly stopped matching the other. It
  lives HERE, next to `astar`, so the twins cannot drift again.

  `radius` is a bound, not a search preference: the halo is at most the
  7-cell inflation, so 10 covers it with margin, while a nearest cell
  farther than that means the map around the robot is unknown -- planning
  from half a metre of teleport would paper over a genuinely lost robot.
  """
  rows, cols = traversable.shape
  sx = min(max(start[0], 0), cols - 1)
  sy = min(max(start[1], 0), rows - 1)
  if traversable[sy, sx]:
    return (sx, sy)
  best, best_d = None, radius * radius + 1
  for dy in range(-radius, radius + 1):
    for dx in range(-radius, radius + 1):
      x2, y2 = sx + dx, sy + dy
      d = dx * dx + dy * dy
      if (d < best_d and 0 <= x2 < cols and 0 <= y2 < rows
          and traversable[y2, x2]):
        best, best_d = (x2, y2), d
  return best


def astar(
  traversable,
  start: tuple[int, int],
  goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
  """Shortest 4-connected path from start to goal, or None if unreachable.

  traversable: 2-D bool array indexed [iy, ix]. start/goal: (ix, iy) cells.
  Returns the path as [(ix, iy), ...] including both endpoints.

  The start cell is always treated as traversable: the robot is physically
  standing on it, whatever the inflated map claims (it may sit inside an
  obstacle's inflation ring after a tight approach and must be able to leave).
  """
  rows, cols = traversable.shape
  gx, gy = goal
  if not (0 <= gx < cols and 0 <= gy < rows) or not traversable[gy, gx]:
    return None
  if start == goal:
    return [start]

  def h(cell):
    return abs(cell[0] - gx) + abs(cell[1] - gy)

  open_heap = [(h(start), 0, start)]     # (f = g + h, g, cell)
  came_from: dict[tuple[int, int], tuple[int, int]] = {}
  best_g = {start: 0}

  while open_heap:
    _, g, cell = heapq.heappop(open_heap)
    if cell == goal:
      path = [cell]
      while cell in came_from:
        cell = came_from[cell]
        path.append(cell)
      return path[::-1]
    if g > best_g.get(cell, g):
      continue                           # stale heap entry; a cheaper route won
    x, y = cell
    for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
      nx, ny = nxt
      if not (0 <= nx < cols and 0 <= ny < rows) or not traversable[ny, nx]:
        continue
      ng = g + 1
      if ng < best_g.get(nxt, float("inf")):
        best_g[nxt] = ng
        came_from[nxt] = cell
        heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))

  return None                            # open set exhausted: no route exists
