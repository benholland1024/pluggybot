"""The store's stock (issue #736, #560's first store task: stocking a shelf,
easy mode): a pillar candle and a desk frame, the cart they wait on, and the
labels that mark the shelf's slots -- as an ACTIVITY (docs/ActivityPattern.md):
their geometry, their hidden parameters and what is sensed of them. Live only
while a job sets them out, as the table's dishware is (`activity.dishware`,
whose put-away this uses): compiled PUT AWAY (`dishware.park_spec`) -- under
the floor, contacts off, a product's weight held, in the geom group no sensor
casts against and no camera draws -- and set out by an offer (`place`).

Each product is shaped for the Dex1-1's two jaws (90 mm) and taken level
across its body (#736's grips, flown on 11-20):

  candle  a cylinder 70 mm across and 100 mm tall, its tag on its top
  frame   a desk frame: a panel 75 mm wide, 115 tall and 20 thick, standing
          on its bottom edge and a strut behind it, its tag on its face
  cart    a top on legs (the dining tables' kind, `home.world.table_geoms`)
          the products wait on: a mocap body a job sets out and puts away
  labels  one tag on each slot's front edge, its top at the board's: a
          mocap body of films, set out and put away with the rest

THE HIDDEN PARAMETERS (`Load`, drawn by `draw`): what each product weighs and
how it slides. ⚠ THE TRUTH IS KEPT AS `Task.secret` IS: never in the robot's
context. Which label each product goes over is the job's word, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from pluggybot.activity import dishware as dw
from pluggybot.activity.base import Activity, Threshold
from pluggybot.perception.tags import (LABEL_TAG_IDS, STOCK_TAG_IDS, STOCK_TAG_SIZES,
                                       plate_half_extent)

#: The products, in the order their tags are numbered.
PRODUCTS = ("candle", "frame")
TAG_OF = dict(zip(PRODUCTS, STOCK_TAG_IDS))
#: The labels, and the board each is on (`home.world.SHELF_BOARDS_Z`'s
#: index: the second and third from the floor, whose fronts the body
#: reaches, #736's call 1), three a board left to right as one faces it.
LABEL_IDS = LABEL_TAG_IDS
BOARD_OF = {i: (1 if k < 3 else 2) for k, i in enumerate(LABEL_IDS)}
#: The bodies a job sets out and puts away: the products, the cart, the labels.
CART, LABELS = "stock_cart", "shelf_labels"
BODIES = (*PRODUCTS, CART, LABELS)

#: THE CANDLE: its radius and height, m: taken across its body, 70 mm of the
#: jaws' 90, as the cup is.
CANDLE_R, CANDLE_H = 0.035, 0.100
#: THE FRAME: its panel's half-extents -- its thickness (its face looks along
#: the body's -x), its width, its height -- m; its strut, behind the panel:
#: its half-extents (thickness, width, length), how far it leans back from
#: upright, rad, and the share of the frame's mass it carries.
PANEL = (0.010, 0.0375, 0.0575)
STRUT = (0.003, 0.012, 0.045)
STRUT_LEAN, STRUT_SHARE = math.radians(25.0), 0.15
#: THE CART: its top's half-extents -- across, out of the shelf, and along
#: it -- m, and its top's height, m: a dining table's height for a 1.4 m body.
#: ⚠ 0.35 m deep, not #736's 0.5: standing 0.13 m off its front, as at the
#: table, the robot's soles are 0.50 m off the shelf's -- where both boards'
#: fronts are reached and their labels read standing (#736's measures) --
#: so a product is a SIDESTEP from its slot (call 4), never a step back the
#: walker barely takes.
CART_HALF, CART_TOP = (0.175, 0.40), 0.72
#: A tag's film over the face it is printed on, m: a visual geom, no contact.
FILM = dw.FILM


@dataclass(frozen=True)
class Load:
  """One job's hidden parameters: what each product weighs (kg) and its
  sliding friction on what it rests on -- and, beside them, which label each
  goes over (`slots`, (product, label) pairs: the job's word, never
  hidden), so a lifecycle's `renew` over a draw's `Load` knows them."""

  candle_kg: float = 0.35
  frame_kg: float = 0.30
  friction: float = 0.7
  slots: tuple = ()


#: The ranges `draw` draws from (#736's table of products).
RANGES = {"candle_kg": (0.25, 0.45), "frame_kg": (0.20, 0.40), "friction": (0.5, 0.9)}
#: `draw(k)`'s stream is this and k: a stream of its own, never the set-out's
#: (`evaluation.counter._drawn`).
DRAW_SEED = 7361


def draw(seed: int) -> Load:
  """Job `seed`'s hidden parameters (seeded): drawn from `RANGES`."""
  if seed < 0:
    raise ValueError(f"a job's seed is a whole number from 0, not {seed}")
  rng = np.random.default_rng((DRAW_SEED, seed))
  return Load(**{n: float(rng.uniform(*r)) for n, r in RANGES.items()})


def _material(spec, tag_id: int) -> str:
  return dw._material(spec, tag_id)


def _free(spec, name: str, at, yaw: float, z: float):
  return dw._free(spec, name, at, yaw, z)


def attach_candle(spec, at, yaw: float, z: float, kg: float, friction: float,
                  tags: bool = True, name: str = "candle"):
  """The candle standing at `at` (x, y) on the surface `z` up, its tag on its
  top."""
  import mujoco
  body = _free(spec, name, at, yaw, z)
  body.add_geom(name=f"{name}_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[CANDLE_R, CANDLE_H / 2, 0], pos=[0, 0, CANDLE_H / 2], mass=float(kg),
                friction=[float(friction), 0.005, 0.0001], rgba=[0.95, 0.92, 0.80, 1])
  dw._film(body, f"{name}_tag", TAG_OF["candle"], STOCK_TAG_SIZES[TAG_OF["candle"]], CANDLE_H,
           spec, tags)
  return body


def strut_pose() -> tuple[list[float], list[float]]:
  """The strut's centre and turn in the frame's body: leaning back
  `STRUT_LEAN` from the panel's back face high up to the surface behind it,
  its lowest edge on the surface -- so the frame stands with its panel
  upright on the panel's bottom edge and the strut's foot."""
  s, c = math.sin(STRUT_LEAN), math.cos(STRUT_LEAN)
  cx = PANEL[0] + STRUT[2] * s
  cz = STRUT[0] * s + STRUT[2] * c
  # turned about y by -lean: its long axis up and toward the panel (-x)
  return [cx, 0.0, cz], [math.cos(-STRUT_LEAN / 2), 0.0, math.sin(-STRUT_LEAN / 2), 0.0]


def attach_frame(spec, at, yaw: float, z: float, kg: float, friction: float,
                 tags: bool = True, name: str = "frame"):
  """The desk frame standing at `at` (x, y) -- its panel's bottom middle --
  on the surface `z` up, its face toward `yaw` + pi (its body's -x), its tag
  on its face."""
  import mujoco
  body = _free(spec, name, at, yaw, z)
  fr = [float(friction), 0.005, 0.0001]
  rgba = [0.45, 0.30, 0.20, 1]
  body.add_geom(name=f"{name}_panel", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(PANEL),
                pos=[0, 0, PANEL[2]], mass=float(kg) * (1 - STRUT_SHARE), friction=fr, rgba=rgba)
  pos, quat = strut_pose()
  body.add_geom(name=f"{name}_strut", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(STRUT), pos=pos,
                quat=quat, mass=float(kg) * STRUT_SHARE, friction=fr, rgba=rgba)
  if tags:
    half = plate_half_extent(STOCK_TAG_SIZES[TAG_OF["frame"]])
    body.add_geom(name=f"{name}_tag", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[FILM / 2, half, half], pos=[-PANEL[0] - FILM / 2, 0.0, PANEL[2]],
                  contype=0, conaffinity=0, mass=0.0, material=_material(spec, TAG_OF["frame"]))
  return body


ATTACH = {"candle": attach_candle, "frame": attach_frame}


def attach_cart(spec, at, yaw: float, name: str = CART):
  """The cart standing at `at` (x, y), turned `yaw` -- its frame's x across
  it, out of the shelf, and y along it -- a mocap body a job sets out and
  puts away: its top and four legs, an inertial frame unturned at its middle
  (a ray's cull, `perception.cull.static_inertials`: a body that moves keeps
  MuJoCo's bug where its frame is turned)."""
  import mujoco
  from pluggybot.home.world import table_geoms
  body = spec.worldbody.add_body(name=name, pos=[float(at[0]), float(at[1]), 0.0],
                                 quat=dw._yaw_quat(yaw), mocap=True)
  for s, c, hh in table_geoms(CART_HALF[0], CART_HALF[1], CART_TOP):
    body.add_geom(name=f"{name}_{s}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(c), size=list(hh),
                  rgba=[0.55, 0.56, 0.60, 1] if s == "geom" else [0.35, 0.36, 0.40, 1])
  body.explicitinertial = True
  body.ipos, body.iquat = [0.0, 0.0, CART_TOP / 2], [1.0, 0.0, 0.0, 0.0]
  body.mass, body.inertia = 8.0, [0.5, 0.5, 0.5]
  return body


def attach_labels(spec, at, yaw: float, places: dict, tags: bool = True, name: str = LABELS):
  """The slots' labels as one mocap body at `at` (x, y), turned `yaw` (its
  -x out of the shelf's front): each label `places[id]` = (along, top) --
  this far along its body's y and its top this high -- a film on the
  board's front edge, looking out of the shelf."""
  import mujoco
  body = spec.worldbody.add_body(name=name, pos=[float(at[0]), float(at[1]), 0.0],
                                 quat=dw._yaw_quat(yaw), mocap=True)
  for i, (along, top) in places.items():
    half = plate_half_extent(STOCK_TAG_SIZES[i])
    body.add_geom(name=f"label_{i}_tag", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[FILM / 2, half, half], pos=[-FILM / 2, float(along), float(top) - half],
                  contype=0, conaffinity=0, mass=0.0,
                  **({"material": _material(spec, i)} if tags else {"rgba": [1, 1, 1, 1]}))
  return body


# ---- set out and put away (the dishware's, #560) --------------------------------


def place(model, data, products: dict, cart: tuple, labels: tuple, z: float = CART_TOP) -> None:
  """The stock SET OUT: each product of `products` ({name: (x, y, yaw)})
  standing on the cart's top, `z` up, the cart at `cart` (x, y, yaw) and the
  labels at `labels` (x, y, yaw) -- all live and still."""
  for name, (x, y, yaw) in products.items():
    b = model.body(name).id
    dw._switch(model, b, True)
    dw._pose(model, data, b, (x, y, z), dw._yaw_quat(yaw))
  for name, (x, y, yaw) in ((CART, cart), (LABELS, labels)):
    b = model.body(name).id
    dw._switch(model, b, True)
    dw._pose(model, data, b, (x, y, 0.0), dw._yaw_quat(yaw))


#: A product reads LIFTED with its bottom this far over where it rested, m,
#: and down again under half of it: the dishware's. Where it belongs is the
#: grader's (`challenge/shelf.py`).
LIFT_ON = dw.LIFT_ON


class Stock(Activity):
  """The products' state machine, off their poses (a camera over the shop,
  never the robot's senses):

    setOut  LIVE: the stock out where a job set it, never put away (its
            cart's geoms out of the parked group, `place` / `park`)
    lifted  LIVE, each product: its bottom more than `LIFT_ON` over where
            it rested when first sensed out, while it is out

  ⚠ `load` is the truth (`draw`): nothing of it is a flag. `slots` is the
  job's word, which label each product goes over -- said to the robot,
  never hidden -- and what the grader judges it against."""

  def __init__(self, model, data, load: Load, slots: dict | None = None,
               name: str = "stock") -> None:
    super().__init__(name)
    from pluggybot.perception.cull import PARKED_GROUP
    self.bodies = {p: int(model.body(p).id) for p in PRODUCTS}
    self._cart_geom = int(model.body_geomadr[model.body(CART).id])
    self._parked = PARKED_GROUP
    self.renew(model, load, slots)

  def renew(self, model, load: Load, slots: dict | None = None) -> None:
    """Begun again over `load` and `slots`, the stock set out afresh: its
    thresholds clear, and where each product rests read again once out."""
    self.load = load
    self.slots = dict(slots if slots is not None else load.slots)
    self.lifted = {p: Threshold(on=LIFT_ON, off=LIFT_ON / 2) for p in PRODUCTS}
    #: The height each product rested at, set out: read at the first sense
    #: that finds it out, off where `place` put it -- never `xpos` (the
    #: dishware's reason: an offer sets out in the physics hook).
    self.rest: dict[str, float] | None = None
    self.set(setOut=bool(model.geom_group[self._cart_geom] != self._parked),
             **{f"{p}Lifted": False for p in PRODUCTS})

  def kept_state(self) -> dict:
    return {**super().kept_state(), "lifted": {p: t.value for p, t in self.lifted.items()},
            "rest": None if self.rest is None else dict(self.rest),
            "slots": dict(self.slots)}

  def restore_kept(self, state: dict) -> None:
    super().restore_kept(state)
    for p, v in (state.get("lifted") or {}).items():
      if p in self.lifted:
        self.lifted[p].value = bool(v)
    if "rest" in state:
      self.rest = None if state["rest"] is None else {p: float(z) for p, z in state["rest"].items()}
    if state.get("slots") is not None:
      self.slots = {p: int(i) for p, i in state["slots"].items()}

  def sense(self, model, data) -> None:
    if model.geom_group[self._cart_geom] == self._parked:
      if self.flags.get("setOut") is not False:
        self.set(setOut=False, **{f"{p}Lifted": False for p in PRODUCTS})
      self.rest = None
      return
    if self.rest is None:
      self.rest = {p: float(dw._placed_at(model, data, b)[2]) for p, b in self.bodies.items()}
    self.set(setOut=True,
             **{f"{p}Lifted": self.lifted[p].update(float(data.xpos[b][2]) - self.rest[p])
                for p, b in self.bodies.items()})
