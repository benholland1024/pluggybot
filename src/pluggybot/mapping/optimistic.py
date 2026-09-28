"""Walking into the unknown (issue #381): a route over the floor the robot
has mapped AND the floor it has not seen yet.

`astar` over `traversable_mask` plans through known-free cells only, and
answers a goal off them with a stand-in: the reachable cell nearest it in a
straight line. From inside a house that cell is against the wall between
the robot and the goal, and the robot stood there. Here a cell nobody has
seen is floor at a price (`UNKNOWN_COST`), a known wall grown by the body's
inflation is not floor at all, and the plan runs as straight as the map
allows. The LIDAR sees the walls it crossed as the robot walks, the next
plan goes round them, and a doorway is found by finding the walls either
side of it. No route at all is then a fact about the map: the walls the
robot has seen enclose the goal, or the robot.

The search is scipy's compiled Dijkstra over an 8-connected lattice of
`BLOCK` x `BLOCK` map cells, built once per grid shape; a plan only writes
its weights (`Lattice`). SimNotes, "Walking into the unknown", has what it
replaced and what it costs.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import distance_transform_cdt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from pluggybot.mapping.astar import nearest_traversable
from pluggybot.mapping.frontier import FREE_THRESH, OCC_THRESH

#: What a metre through space the robot has not seen costs, against a metre
#: of floor it has mapped: a known detour is taken while it is under twice
#: the unknown shortcut. A CHOICE, not a fit: swept 1.25 / 2 / 4 over six
#: fresh walks (`scripts/unknown_spike.py --unknown-cost`) it moved four not
#: at all and two by 3 s -- a fresh map offers no known detour to weigh.
UNKNOWN_COST = 2.0
#: The lattice's cell, in map cells: 10 cm at the map's 5 cm. A lattice cell
#: passes only if every map cell in it does, so a gap keeps a lattice path
#: while it is 3 map cells wide -- a 1 m door, less 0.35 m of inflation a
#: side, is 6. MEASURED at 5 cm on the home grid, the same search cost ~5x
#: as much (~260 ms against ~60 ms a plan, busy box).
BLOCK = 2
#: The halo escape, in lattice cells: the map's 10 (`nearest_traversable`).
ESCAPE_CELLS = 5
#: A goal on the floor is searched for first within the straight line to it
#: at the dearest cost and this much more, m (`route`).
LIMIT_SLACK_M = 2.0
#: The lattice's eight steps, (dx, dy).
STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def map_costs(logodds: np.ndarray, inflation_cells: int,
              unknown_cost: float = UNKNOWN_COST) -> np.ndarray:
  """Per map cell, what a metre through it costs: 1 over known-free floor,
  `unknown_cost` where nothing has been seen, and inf in a wall or within
  `inflation_cells` of one (`traversable_mask`'s taxicab inflation, grown
  into unknown space too: the far side of a wall is not floor)."""
  occupied = logodds > OCC_THRESH
  cost = np.where(logodds < FREE_THRESH, 1.0, float(unknown_cost))
  if occupied.any():
    cost[distance_transform_cdt(~occupied, metric="taxicab") <= inflation_cells] = np.inf
  return cost


def coarsen(cost: np.ndarray, block: int = BLOCK) -> np.ndarray:
  """The lattice's cells: each `block` x `block` map cells, passing only if
  every one of them does, at the dearest of their costs. A grid whose sides
  are not a multiple of `block` is padded with wall."""
  rows, cols = cost.shape
  r, c = -(-rows // block), -(-cols // block)
  if (r * block, c * block) != (rows, cols):
    padded = np.full((r * block, c * block), np.inf)
    padded[:rows, :cols] = cost
    cost = padded
  # strided maxima, not a max over a 4-d view: ~8x cheaper on home's grid
  out = cost[0::block, 0::block].copy()
  for dy in range(block):
    for dx in range(block):
      if dy or dx:
        np.maximum(out, cost[dy::block, dx::block], out=out)
  return out


class Lattice:
  """The 8-connected graph of a `rows` x `cols` lattice as one CSR matrix,
  built once: a plan writes its weights and runs Dijkstra. Every node has
  eight slots, one a step (`STEPS`), and a slot off the lattice's edge is a
  loop that stays inf. A diagonal step may not cut a corner -- both cells
  it squeezes between must pass -- so the lattice's components are its
  4-connected ones."""

  def __init__(self, rows: int, cols: int, cell_m: float) -> None:
    self.shape, self.cell_m = (rows, cols), float(cell_m)
    n, k = rows * cols, len(STEPS)
    idx = np.arange(n, dtype=np.int32).reshape(rows, cols)
    to = np.repeat(idx[:, :, None], k, axis=2)
    for i, (dx, dy) in enumerate(STEPS):
      ys = slice(max(0, -dy), rows - max(0, dy))
      xs = slice(max(0, -dx), cols - max(0, dx))
      to[ys, xs, i] = idx[ys.start + dy:ys.stop + dy, xs.start + dx:xs.stop + dx]
    self.graph = csr_matrix((np.full(n * k, np.inf), to.ravel(),
                             np.arange(0, n * k + 1, k, dtype=np.int32)),
                            shape=(n, n))
    self._slots = self.graph.data.reshape(rows, cols, k)
    # a step's weight is the mean of its two cells' costs over its length
    self._half = [0.5 * self.cell_m * math.hypot(dx, dy) for dx, dy in STEPS]
    # the costs with a wall round them, and each step's weights, contiguous:
    # written a step at a time and interleaved into the slots in one copy --
    # MEASURED, writing each step straight into its stride-8 slots cost
    # 2.3x as much (15.2 ms against 6.5 on home's lattice)
    self._pad = np.full((rows + 2, cols + 2), np.inf)
    self._steps = np.empty((k, rows, cols))

  def weigh(self, cost: np.ndarray) -> None:
    """Write every step's weight off the lattice's per-cell costs: inf
    wherever either end -- or, for a diagonal, either cell beside it --
    does not pass (scipy's Dijkstra takes an inf edge as none)."""
    rows, cols = self.shape
    pad = self._pad
    pad[1:-1, 1:-1] = cost
    blocked = np.where(np.isfinite(pad), 0.0, np.inf)
    here = pad[1:-1, 1:-1]
    for i, (dx, dy) in enumerate(STEPS):
      w = self._steps[i]
      np.add(here, pad[1 + dy:rows + 1 + dy, 1 + dx:cols + 1 + dx], out=w)
      w *= self._half[i]
      if dx and dy:
        w += blocked[1:-1, 1 + dx:cols + 1 + dx]
        w += blocked[1 + dy:rows + 1 + dy, 1:-1]
    np.copyto(self._slots, np.moveaxis(self._steps, 0, -1))

  def search(self, source: int, limit: float = np.inf) -> tuple[np.ndarray, np.ndarray]:
    """Dijkstra over the weights last written (`weigh`) from one lattice
    index, no further than `limit`: (the cost to every cell, inf where none
    reaches within it; each cell's predecessor, negative at the source and
    where none reaches)."""
    return dijkstra(self.graph, directed=True, indices=source,
                    return_predecessors=True, limit=limit)


_LATTICES: dict = {}


def lattice_for(shape: tuple[int, int], cell_m: float) -> Lattice:
  """The lattice of a shape, built once a process (~0.1-0.4 s at home's
  230 x 510): every robot planning over a grid of that shape shares it."""
  key = (int(shape[0]), int(shape[1]), round(float(cell_m), 9))
  if key not in _LATTICES:
    _LATTICES[key] = Lattice(*key)
  return _LATTICES[key]


def route(lattice: Lattice, cost: np.ndarray, start: tuple[int, int],
          goal: tuple[int, int],
          escape: int = ESCAPE_CELLS) -> tuple[list[tuple[int, int]] | None, bool]:
  """The cheapest lattice path from `start` to `goal`, cells as (ix, iy)
  from the first cell it stands on to the last, and whether that last is a
  STAND-IN; or (None, False).

  `start` off the lattice's floor escapes to the nearest cell that is on it
  within `escape` (the halo escape, `nearest_traversable`), and none there
  is no route: the robot is lost, not parked by a wall. A goal on the floor
  and reachable is the end; on the floor and NOT reachable is no route
  (another robot's disc, or walls, between); OFF the floor -- in a wall or
  its inflation -- the end is its stand-in, the reachable cell nearest it
  in a straight line (the cheaper of two equally near).

  A goal on the floor is searched for no further than the straight line to
  it through the unknown and `LIMIT_SLACK_M` (the search is exact within
  its limit, so the path is the same), and past that without one: most
  walks are a few metres, and a search stopped there costs a fraction of
  one across the whole lattice."""
  rows, cols = cost.shape
  passes = np.isfinite(cost)
  here = nearest_traversable(passes, start, radius=escape)
  if here is None:
    return None, False
  source = here[1] * cols + here[0]
  lattice.weigh(cost)
  gx = min(max(int(goal[0]), 0), cols - 1)
  gy = min(max(int(goal[1]), 0), rows - 1)
  end = gy * cols + gx
  stand_in = not passes[gy, gx]
  if stand_in:
    dist, pred = lattice.search(source)
    reached = np.flatnonzero(np.isfinite(dist))
    ry, rx = np.divmod(reached, cols)
    near = (rx - gx) ** 2 + (ry - gy) ** 2
    nearest = reached[near == near.min()]
    end = int(nearest[np.argmin(dist[nearest])])
  else:
    straight = math.hypot(gx - here[0], gy - here[1]) * lattice.cell_m
    dist, pred = lattice.search(source, limit=float(cost[passes].max()) * straight
                                + LIMIT_SLACK_M)
    if not np.isfinite(dist[end]):
      dist, pred = lattice.search(source)
    if not np.isfinite(dist[end]):
      return None, False
  cells = []
  while end != source:
    cells.append((end % cols, end // cols))
    end = int(pred[end])
  cells.append(here)
  return cells[::-1], stand_in
