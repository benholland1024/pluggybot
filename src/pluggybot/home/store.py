"""The store's back shelf (issue #736, #560's first store task: stocking a
shelf, easy mode): where its slots' labels are on the shop's shelf
(`home.world`'s `shop_shelf`), where the cart stands beside them, and the
stock -- a candle, a desk frame, the cart and the labels (`activity/stock.py`)
-- put into a world's spec PUT AWAY until a job sets it out, as the
restaurant's dishware is (`home.restaurant`): every free body costs the
physics and every geom a ray, and a world is compiled once.

⚠ NOT IN THE SERVED HOUSE YET (#736's stage 1, the loop of 2026-11-22):
`humanoid.world.home_spec` does not call `attach`. Every body it puts in
moves the served world's geometry fingerprint (`continuation.fingerprint`:
both robots' maps and bodies are dropped at the deploy that brings it) and
the scenes the site vendors (`telemetry.scene`), so it comes in with the job
that offers it. The offline check (`evaluation.counter.shelf_set_out`) and
`scripts/solve.py --feature shelf` put it into their own house.

The shelf's own frame here, S: from the middle of its boards' front edge on
the floor, `s` along it to the right as one faces it and `d` out of its
front toward the aisle. Nothing here is the robot's: it finds the slots by
their labels (79-84) and the products by theirs (76, 77).

The restaurant's members (`home.jobs`): a lifecycle sets it out for an
offer, puts it back after a restart and keeps an offer off it while a
robot touches it -- in a world that has it (`present`), which no served one
does yet; `stock_shelf` is a kind no cadence names (`economy.tasks`).
"""

from __future__ import annotations

import math

import numpy as np

from pluggybot.activity import dishware as dw
from pluggybot.activity import stock as st
from pluggybot.home import world as home

#: The shelf the job stocks (`home.world.SHELVES`), and its entry: its
#: middle, its length and depth, m, and its turn (its front faces its
#: frame's +y turned by it).
SHELF = "shop_shelf"
_ENTRY = next(s for s in home.SHELVES if s[0] == SHELF)
SHELF_XY, SHELF_LEN, SHELF_DEPTH, SHELF_YAW = (_ENTRY[1], _ENTRY[2]), _ENTRY[3], _ENTRY[4], _ENTRY[5]
#: The job's target: the shelf's own body, and a `shelf`.
TARGETS = {SHELF: "shelf"}
#: ...and whether a served house has it (`home.jobs.targets`): not until the
#: release that puts `attach` into `home_spec` (this module's docstring).
SERVED = False
#: The slots, along the shelf: the middle one this far to the right of the
#: shelf's middle, m, and the others this far either side of it -- three a
#: board, each label at its slot's middle (#736's call 3), the products 7-8
#: cm wide.
SLOTS_AT, SLOT_GAP = 0.40, 0.25
#: The cart (#736's call 4, easy mode): its middle this far along (to the
#: left of the slots), its back this far out of the shelf's front, m -- at
#: the shelf beside the slots, a sidestep a product.
CART_AT, CART_BACK_GAP = -0.45, 0.02


def axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """The shelf's frame S in the world, off its drawing: its origin (x, y),
  the way out of its front, and the way to the right as one faces it."""
  out = np.array([-math.sin(SHELF_YAW), math.cos(SHELF_YAW)])
  right = np.array([-out[1], out[0]])
  return np.array(SHELF_XY) + out * SHELF_DEPTH / 2, out, right


def to_world(s: float, d: float) -> tuple[float, float]:
  """A point of S in the world: `s` along to the right, `d` out of the front."""
  origin, out, right = axes()
  x, y = origin + s * right + d * out
  return float(x), float(y)


def facing_in() -> float:
  """The heading of one facing the shelf, rad: into it."""
  _, out, _ = axes()
  return math.atan2(-out[1], -out[0])


def board_top(board: int) -> float:
  """A board's top over the floor, m."""
  return home.SHELF_BOARDS_Z[board] + home.SHELF_T / 2


def label_along(label: int) -> float:
  """A label's middle along the shelf, `s`, m: left to right as one faces it."""
  k = st.LABEL_IDS.index(label) % 3
  return SLOTS_AT + (k - 1) * SLOT_GAP


def label_places() -> dict[int, tuple[float, float]]:
  """Each label in the labels' body (`stock.attach_labels`): (its middle
  along the body's y, its top), the body standing at S's origin facing in --
  its y the left hand's way, so a label `s` to the right is at -s."""
  return {i: (-label_along(i), board_top(st.BOARD_OF[i])) for i in st.LABEL_IDS}


def cart_place() -> tuple[float, float, float]:
  """The cart's middle (x, y) and its turn: facing in, its x out of the shelf."""
  x, y = to_world(CART_AT, CART_BACK_GAP + st.CART_HALF[0])
  return x, y, facing_in()


def labels_place() -> tuple[float, float, float]:
  """The labels' body (x, y) and its turn: at S's origin, facing in."""
  x, y = to_world(0.0, 0.0)
  return x, y, facing_in()


def attach(spec, so, parked: bool = True) -> None:
  """The stock into a world's `spec`: the cart, the labels and each product
  where the set-out `so` (`counter.shelf_set_out`) puts it, at its hidden
  weights and friction -- PUT AWAY (`dishware.park_spec`) unless `parked`
  is False, as an offline set-out's own world has it out."""
  cx, cy, cyaw = cart_place()
  lx, ly, lyaw = labels_place()
  bodies = {st.CART: st.attach_cart(spec, (cx, cy), cyaw),
            st.LABELS: st.attach_labels(spec, (lx, ly), lyaw, label_places())}
  for name, (x, y, yaw) in so.products.items():
    bodies[name] = st.ATTACH[name](spec, (x, y), yaw, st.CART_TOP,
                                   getattr(so.load, f"{name}_kg"), so.load.friction)
  if parked:
    # ...by the bodies themselves: a house with a robot in it finds none of
    # them by name until it is compiled (`dishware.park_body`)
    for name, body in bodies.items():
      dw.park_body(body, PARK_AT[name])


def _middles() -> dict:
  """Where each body waits put away, (x, y): under the cart's middle."""
  x, y, _ = cart_place()
  return {name: (x, y) for name in st.BODIES}


PARK_AT = _middles()


def present(model) -> bool:
  """Whether this world has the stock, put away to be set out."""
  try:
    return int(model.body(st.CART).mocapid[0]) >= 0
  except KeyError:
    return False


def _ours(target: str) -> None:
  if target not in TARGETS:
    raise KeyError(target)


def drawn(target: str, k: int) -> st.Load:
  """The stock's hidden parameters at draw `k` (`stock.draw`) -- kept as
  `Task.secret` is, never a flag -- with the labels set-out k names."""
  from dataclasses import replace
  _ours(target)
  return replace(st.draw(k), slots=tuple(sorted(labels(k).items())))


def words(target: str, k: int) -> dict[str, int]:
  """What an offer at draw `k` names (`home.jobs.words`): its labels."""
  _ours(target)
  return labels(k)


def labels(k: int) -> dict[str, int]:
  """Which label each product goes over at draw `k`, as its offer says it
  (`{candle}`, `{frame}`): set-out k's."""
  from pluggybot.evaluation import counter as ct
  return dict(ct.shelf_set_out(k).slots)


def redraw(model, data, target: str, k: int) -> None:
  """The products' hidden weights and friction at draw `k` set into the
  compiled world -- each one's mass and inertia (its shape's, scaled), every
  solid geom of it the drawn friction -- the derived constants recomputed
  (`dishware.set_const`, which counts the products' hold too)."""
  _ours(target)
  load = st.draw(k)
  for p in st.PRODUCTS:
    b = model.body(p).id
    kg = float(getattr(load, f"{p}_kg"))
    model.body_inertia[b] *= kg / float(model.body_mass[b])
    model.body_mass[b] = kg
    for g in range(int(model.body_geomadr[b]), int(model.body_geomadr[b] + model.body_geomnum[b])):
      if not dw._is_film(model, g):
        model.geom_friction[g][0] = float(load.friction)
  dw.set_const(model)


def set_out(model, data, target: str, k: int, kind: str = "shelf") -> None:
  """The stock set out for draw `k`: the products where set-out k puts them
  on the cart, the cart and the labels at the shelf, live and still, the
  hidden parameters drawn (`redraw`)."""
  from pluggybot.evaluation import counter as ct
  _ours(target)
  so = ct.shelf_set_out(k)
  redraw(model, data, target, k)
  st.place(model, data, so.products, cart_place(), labels_place())


def put_away(model, data, target: str) -> None:
  """The stock parked again: no job holds the shelf."""
  _ours(target)
  dw.park(model, data, PARK_AT)


def away(model, data, target: str) -> bool:
  """Whether the stock is put away where `put_away` puts it."""
  _ours(target)
  return dw.parked(model, data, PARK_AT)


def activity_for(model, data, target: str, k: int):
  """The stock's activity over draw `k`, named by its shelf
  (`stock.Stock`), its slots set-out k's."""
  from pluggybot.evaluation import counter as ct
  _ours(target)
  return st.Stock(model, data, st.draw(k), ct.shelf_set_out(k).slots, name=target)


def activities(model, data) -> list:
  """The stock's activity where the world has it, or none."""
  return [activity_for(model, data, SHELF, 0)] if present(model) else []


def touched_by(model, data, target: str, geoms) -> bool:
  """Whether anything of `geoms` (a robot's) touches a product or the cart:
  stock a hand is on is not set out or put away from under it."""
  from pluggybot.contacts import contact_pairs
  _ours(target)
  own = np.isin(model.geom_bodyid, [model.body(n).id for n in st.BODIES])
  g = contact_pairs(data)
  if not g.shape[0]:
    return False
  other = np.isin(np.arange(model.ngeom), list(geoms))
  a, b = g[:, 0], g[:, 1]
  return bool(((own[a] & other[b]) | (own[b] & other[a])).any())
