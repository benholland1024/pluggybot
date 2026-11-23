"""Stocking a shelf (issue #736, #560's first store task, easy mode): what it
stands on, each pinned without a flight (docs/Testing.md).

  1. the grader (`challenge/shelf.py`) passes only what its criteria say,
     each criterion failing alone for its own reason, judged unrounded and
     shown rounded, a fall first, and a reading past its line never printed
     as the line's own number;
  2. the shelf's frame is its boards', and a product is measured in it
     against its label: along the board, over its top, its front, its tilt
     and the frame's turn;
  3. a product the robot holds is never dropped: only a free product's
     fall is one (the table's watch, over the products);
  4. the products stand where they are set down, the frame on its strut;
  5. the set-out is seeded, its products on the cart inside their ranges,
     each its own label, the hidden draw a stream of its own;
  6. the stock is put away as the dishware is, set out and put away again,
     and the products' hold survives a redraw (`dishware.set_const`);
  7. the tags are the registry's, and the row pays nothing, shown to no mind.
"""

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from pluggybot.activity import dishware as dw
from pluggybot.activity import stock as st
from pluggybot.challenge import lid, shelf
from pluggybot.economy import scoring
from pluggybot.evaluation import counter as ct
from pluggybot.home import store
from pluggybot.home import world as home
from pluggybot.perception import tags

SRC = Path(__file__).parent.parent / "src" / "pluggybot"
BARE = """<mujoco model="shelf"><option integrator="implicitfast"/>
  <worldbody><geom name="floor" type="plane" size="40 40 0.1"/></worldbody></mujoco>"""
SLOTS = {"candle": 80, "frame": 83}


# ---- the grader: each criterion fails alone -----------------------------------------


def _stocked(**kw):
  product = {"slot": 0, "liftedM": 0.05, "fastestFreeMS": 0.1, "alongAtCallM": 0.01,
             "alongM": 0.01, "aboveM": 0.0, "backM": 0.04, "tiltDeg": 1.0}
  m = {"robotTiltDeg": 5.0, "falls": 0, "touchedDuringHold": [], "holdS": shelf.HOLD_S,
       "elapsedS": 120.0, "limitS": 300.0,
       **{p: {**product, "slot": SLOTS[p]} for p in st.PRODUCTS}}
  m["frame"]["turnDeg"] = 3.0
  for k, v in kw.items():
    if isinstance(v, dict):
      m[k] = {**m[k], **v}
    else:
      m[k] = v
  return m


def test_the_shelf_passes_stocked_let_go_in_time_and_on_nothing_less():
  ok, metrics, why = shelf.eval_shelf(_stocked())
  assert ok and why == ("the shelf stocked in 120 s -- the candle 1.0 cm and the frame 1.0 cm "
                        "from its label -- and let go"), why
  assert metrics["candle"]["slot"] == 80 and metrics["frame"]["slot"] == 83
  for p in st.PRODUCTS:
    label = SLOTS[p]
    # 1: its middle along the board, either way
    ok, _, why = shelf.eval_shelf(_stocked(**{p: {"alongM": shelf.ALONG_M + 0.006}}))
    assert not ok and why.startswith(f"the {p} ended 5 cm right of label {label}: it must rest "
                                     "on its label's board"), why
    ok, _, why = shelf.eval_shelf(_stocked(**{p: {"alongM": -0.07}}))
    assert not ok and why.startswith(f"the {p} ended 7 cm left of label {label}"), why
    # ...on its board: over it (on another, or on the cart) or under it
    ok, _, why = shelf.eval_shelf(_stocked(**{p: {"aboveM": 0.40}}))
    assert not ok and why.startswith(f"the {p} ended 1 cm right of label {label}, 40.0 cm over "
                                     "its board's top"), why
    ok, _, why = shelf.eval_shelf(_stocked(**{p: {"aboveM": -0.006}}))
    assert not ok and ", 0.6 cm under its board's top" in why, why
    # ...its front within 10 cm of the edge, and not over it
    ok, _, why = shelf.eval_shelf(_stocked(**{p: {"backM": shelf.FRONT_IN_M + 0.02}}))
    assert not ok and why == (f"the {p}'s front ended 12 cm behind the board's edge: its front "
                              "must be within 10 cm of the board's edge and not over it"), why
    ok, _, why = shelf.eval_shelf(_stocked(**{p: {"backM": -0.012}}))
    assert not ok and why.startswith(f"the {p} juts 1.2 cm out over the board's edge"), why
    # ...upright
    ok, _, why = shelf.eval_shelf(_stocked(**{p: {"tiltDeg": math.degrees(shelf.UPRIGHT_RAD) + 3}}))
    assert not ok and why == (f"the {p} stands tipped 13 deg: it must stand upright, within "
                              "10 deg"), why
    # 2: dropped -- a free fall, however it ended
    ok, _, why = shelf.eval_shelf(_stocked(**{p: {"fastestFreeMS": shelf.DROP_M_S + 0.3}}))
    assert not ok and why.startswith(f"the {p} came down at 0.8 m/s with nothing holding it"), why
  # ...and the frame's face toward the aisle, either way
  for turn in (25.0, -25.0):
    ok, _, why = shelf.eval_shelf(_stocked(frame={"turnDeg": turn}))
    assert not ok and why == ("the frame faces 25 deg off the aisle: its face must look out of "
                              "the shelf within 20 deg"), why
  # ...a candle turned any way is a candle
  assert shelf.eval_shelf(_stocked(candle={"turnDeg": 120.0}))[0]
  # 3: let go
  ok, _, why = shelf.eval_shelf(_stocked(touchedDuringHold=["left_jaw_a_pad"]))
  assert not ok and why == ("the shelf stocked, but left_jaw_a_pad still touched it after the "
                            "call -- let go"), why
  # 4: in time, and the hold
  ok, _, why = shelf.eval_shelf(_stocked(elapsedS=301.0))
  assert not ok and why.startswith("called 301 s after the start, past the 300 s the job allows")
  ok, _, why = shelf.eval_shelf(_stocked(holdS=1.0))
  assert not ok and why.startswith("measured 1.0 s after the call")


def test_each_line_is_its_own_and_a_product_on_it_passes():
  """Each line passes on its own side: a product 4 cm along, 4 mm over or
  under its board, its front at the edge or 10 cm behind it, tipped 10 deg,
  the frame turned 20 deg -- and fails just past it."""
  edge = [{"alongM": shelf.ALONG_M}, {"alongM": -shelf.ALONG_M}, {"aboveM": shelf.RESTING_M},
          {"aboveM": -shelf.RESTING_M}, {"backM": 0.0}, {"backM": shelf.FRONT_IN_M},
          {"tiltDeg": math.degrees(shelf.UPRIGHT_RAD)}]
  for p in st.PRODUCTS:
    for e in edge:
      assert shelf.eval_shelf(_stocked(**{p: e}))[0], (p, e)
  assert shelf.eval_shelf(_stocked(frame={"turnDeg": math.degrees(shelf.FACING_RAD)}))[0]
  past = [{"alongM": shelf.ALONG_M + 1e-4}, {"aboveM": shelf.RESTING_M + 1e-5},
          {"backM": -1e-4}, {"backM": shelf.FRONT_IN_M + 1e-4},
          {"tiltDeg": math.degrees(shelf.UPRIGHT_RAD) + 0.01}]
  for p in st.PRODUCTS:
    for e in past:
      assert not shelf.eval_shelf(_stocked(**{p: e}))[0], (p, e)
  assert not shelf.eval_shelf(_stocked(frame={"turnDeg": math.degrees(shelf.FACING_RAD) + 0.01}))[0]


def test_a_reading_past_its_line_never_reads_as_the_line():
  """#737's rule in this grader's words (`lid.against`, the five graders'):
  a reading just past its line is printed at the precision that tells it
  from the line -- "4.01 cm", never "4 cm" against "within 4 cm"."""
  ok, _, why = shelf.eval_shelf(_stocked(candle={"alongM": 0.0401}))
  assert not ok and "ended 4.01 cm right of label 80" in why, why
  ok, _, why = shelf.eval_shelf(_stocked(frame={"backM": 0.1004}))
  assert not ok and "front ended 10.04 cm behind" in why, why
  ok, _, why = shelf.eval_shelf(_stocked(candle={"fastestFreeMS": 0.504}))
  assert not ok and "came down at 0.504 m/s" in why, why
  ok, _, why = shelf.eval_shelf(_stocked(frame={"turnDeg": 20.3}))
  assert not ok and "faces 20.3 deg off the aisle" in why, why
  ok, _, why = shelf.eval_shelf(_stocked(candle={"tiltDeg": 10.2}))
  assert not ok and "tipped 10.2 deg" in why, why
  ok, _, why = shelf.eval_shelf(_stocked(candle={"backM": -0.0003}))
  assert not ok and "juts 0.03 cm out over" in why, why
  # ...and a passing distance said beside a failing height (review round 2)
  ok, _, why = shelf.eval_shelf(_stocked(candle={"alongM": 0.0396, "aboveM": 0.006}))
  assert not ok and why.startswith("the candle ended 3.96 cm right of label 80, 0.6 cm over"), why
  # ...through the five graders' one helper, #737's `lid.against`
  assert lid.against(4.6, 4.0) == ("5", "4")


def test_the_reason_names_every_product_at_fault_once():
  """As the table's (#687): every product at fault, once, by its first
  fault in the criteria's order, each rule once over the products it names,
  a fall first -- and the longest a settled job's History line holds."""
  rule = "must rest on its label's board, its middle within 4 cm of the label's along it"
  ok, _, why = shelf.eval_shelf(_stocked(candle={"alongM": 0.06}, frame={"alongM": -0.3,
                                                                         "aboveM": -0.4}))
  assert not ok and why == ("the candle ended 6 cm right of label 80 and the frame ended 30 cm "
                            "left of label 83, 40.0 cm under its board's top: each " + rule), why
  ok, _, why = shelf.eval_shelf(_stocked(candle={"backM": 0.15}, frame={"backM": -0.02}))
  assert why == ("the candle's front ended 15 cm behind the board's edge and the frame juts 2.0 "
                 "cm out over the board's edge: each one's front must be within 10 cm of the "
                 "board's edge and not over it"), why
  # ...where a product ended before how it stands, a drop before all
  ok, _, why = shelf.eval_shelf(_stocked(
    falls=1, elapsedS=301.0, touchedDuringHold=["left_jaw_a_pad"],
    candle={"fastestFreeMS": 2.1, "alongM": 0.5}, frame={"tiltDeg": 80.0, "alongM": 0.2}))
  assert not ok and why == (
    "you fell, and a fall fails the try -- called 301 s after the start, past the 300 s the job "
    "allows; the candle came down at 2.1 m/s with nothing holding it -- dropped or knocked over, "
    "not set down (no faster than 0.5); the frame ended 20 cm right of label 83: it " + rule), why
  from pluggybot.lifecycle import HISTORY_STAMP_ROOM
  from pluggybot.mind.thoughts import MAX_LINE_CHARS
  _, _, why = shelf.eval_shelf(_stocked(
    falls=2, elapsedS=1001.0,
    candle={"fastestFreeMS": 4.637, "alongM": 0.61, "aboveM": -0.72},
    frame={"fastestFreeMS": 3.2, "alongM": 0.81, "aboveM": -0.72}))
  assert len(f"failed the job t_0042 (stock_shelf): {why}") <= MAX_LINE_CHARS - HISTORY_STAMP_ROOM
  _, _, why = shelf.eval_shelf(_stocked(
    falls=2, elapsedS=1001.0, candle={"alongM": 0.61, "aboveM": -0.72},
    frame={"alongM": -0.812, "aboveM": 0.163}))
  assert len(f"failed the job t_0042 (stock_shelf): {why}") <= MAX_LINE_CHARS - HISTORY_STAMP_ROOM


def test_the_shelf_is_judged_as_measured_shown_rounded_and_a_fall_first():
  ok, metrics, _ = shelf.eval_shelf(_stocked(candle={"alongM": 0.04004}))
  assert not ok and metrics["candle"]["alongM"] == 0.04
  ok, _, why = shelf.eval_shelf(_stocked(falls=1))
  assert not ok and why.startswith("you fell, and a fall fails the try -- the shelf stocked")
  assert shelf.eval_shelf(_stocked(falls=None))[::2] == (False, "your falls were never counted")
  # a missing measurement is not a passing one -- a product with no label
  # named is never measured against one
  for p in st.PRODUCTS:
    for k in ("slot", "fastestFreeMS", "alongM", "aboveM", "backM", "tiltDeg"):
      assert shelf.eval_shelf(_stocked(**{p: {k: None}}))[::2] == (
        False, "the shelf was never measured"), (p, k)
  assert not shelf.eval_shelf(_stocked(frame={"turnDeg": None}))[0]
  # ...and a tilted body is judged by nothing
  assert shelf.eval_shelf(_stocked(robotTiltDeg=80.0))[0]


# ---- the shelf's frame and a product measured in it --------------------------------------


def _shelf_world(at=None, settle_s: float = 0.0):
  """The shop's shelf as the house draws it, its labels, and the products --
  each over its label in `SLOTS` with its front 5 cm behind the board's
  edge, or at `at` (product -> (s, d, z, yaw) in the shelf's frame S, the
  heading off facing in) -- on a bare floor, no robot."""
  spec = mujoco.MjSpec.from_string(BARE)
  name, cx, cy, length, depth, yaw, double = next(s for s in home.SHELVES if s[0] == shelf.SHELF)
  body = spec.worldbody.add_body(name=name, pos=[cx, cy, 0.0],
                                 quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
  for s, c, hh in home.shelf_geoms(length, depth, double):
    body.add_geom(name=f"{name}_{s}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(c), size=list(hh))
  lx, ly, lyaw = store.labels_place()
  st.attach_labels(spec, (lx, ly), lyaw, store.label_places(), tags=False)
  front = {"candle": st.CANDLE_R, "frame": st.PANEL[0]}
  for p in st.PRODUCTS:
    if at and p in at:
      s, d, z, turn = at[p]
    else:
      label = SLOTS[p]
      s, d, z, turn = (store.label_along(label), -0.05 - front[p],
                       store.board_top(st.BOARD_OF[label]), 0.0)
    st.ATTACH[p](spec, store.to_world(s, d), store.facing_in() + turn, z, 0.3, 0.7, tags=False)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  for _ in range(int(settle_s / model.opt.timestep)):
    mujoco.mj_step(model, data)
  return model, data


def _measured(model, data, slots=SLOTS):
  return shelf.measurements(shelf.measure(model, data), shelf.measure(model, data), None, slots)


def _judged(got):
  """A world's measured products judged, the rest of the attempt as `_stocked`'s."""
  base = _stocked()
  return shelf.eval_shelf({**base, **{p: {**base[p], **{k: v for k, v in got[p].items()
                                                         if v is not None}}
                                      for p in st.PRODUCTS}})


def test_a_product_over_its_label_reads_nothing_off_and_each_way_reads_its_own():
  model, data = _shelf_world()
  origin, out, right, tops = shelf.shelf_frame(model, data)
  # the shop's shelf faces -x: facing it, the right hand is -y
  assert np.allclose(out, [-1, 0, 0]) and np.allclose(right, [0, -1, 0])
  assert origin[0] == pytest.approx(home.BACK_X - home.WALL_HALF_T - 0.4)
  assert tops[1] == pytest.approx(0.56) and tops[2] == pytest.approx(0.96)
  got = _measured(model, data)
  for p in st.PRODUCTS:
    assert got[p]["alongM"] == pytest.approx(0.0, abs=1e-9), p
    assert got[p]["aboveM"] == pytest.approx(0.0, abs=1e-9), p
    assert got[p]["backM"] == pytest.approx(0.05, abs=1e-9), p
    assert got[p]["tiltDeg"] == pytest.approx(0.0, abs=1e-6), p
  assert got["frame"]["turnDeg"] == pytest.approx(0.0, abs=1e-6)
  # ...3 cm to the right, 2 cm over the edge, on the board above, turned
  label_c, label_f = SLOTS["candle"], SLOTS["frame"]
  model, data = _shelf_world(at={
    "candle": (store.label_along(label_c) + 0.03, 0.02 - st.CANDLE_R, store.board_top(2), 0.0),
    "frame": (store.label_along(label_f) - 0.02, -0.05 - st.PANEL[0], store.board_top(1), 0.5)})
  got = _measured(model, data)
  assert got["candle"]["alongM"] == pytest.approx(0.03) and got["candle"]["backM"] == pytest.approx(-0.02)
  assert got["candle"]["aboveM"] == pytest.approx(0.40)
  assert got["frame"]["alongM"] == pytest.approx(-0.02) and got["frame"]["aboveM"] == pytest.approx(-0.40)
  assert got["frame"]["turnDeg"] == pytest.approx(math.degrees(0.5))
  # the frame turned 0.5 rad: its front is its panel's face's nearer corner
  assert got["frame"]["backM"] == pytest.approx(0.05 + st.PANEL[0]
                                                - st.PANEL[0] * math.cos(0.5)
                                                - st.PANEL[1] * math.sin(0.5), abs=1e-9)
  ok, _, why = _judged(got)
  assert not ok and why.startswith("the candle ended 3 cm right of label 80, 40.0 cm over its board"), why
  # ...and a product measured against another label is measured against that one
  other = _measured(model, data, {"candle": 84, "frame": 83})
  assert other["candle"]["alongM"] == pytest.approx(0.03 - 0.25) and other["candle"]["aboveM"] == pytest.approx(0.0)


def test_the_products_stand_where_they_are_set_down_the_frame_on_its_strut():
  """The candle and the frame set on a board stand still where they are
  set, upright: the frame's panel and its strut's foot both on the board
  (`stock.strut_pose`), so its rest reads untipped."""
  model, data = _shelf_world(settle_s=1.0)
  up = np.array([0.0, 0.0, 1.0])
  low = {g: data.geom_xpos[model.geom(g).id][2] - shelf._extent(model, data, model.geom(g).id, up)
         for g in ("frame_panel", "frame_strut")}
  assert low["frame_strut"] == pytest.approx(low["frame_panel"], abs=2e-4), low
  got = _measured(model, data)
  for p in st.PRODUCTS:
    assert got[p]["tiltDeg"] < 0.5 and abs(got[p]["aboveM"]) < 0.001, (p, got[p])
    v = data.qvel[model.jnt_dofadr[model.joint(f"{p}_free").id]:][:6]
    assert float(np.abs(v).max()) < 0.01, p
  assert _judged(got)[0]


# ---- a held product is never dropped --------------------------------------------------------


def test_a_product_the_robot_holds_is_carried_down_never_dropped(monkeypatch):
  """The table's line, over the products: only a product nothing of the
  robot touches can be dropped."""
  from pluggybot.body import StubBody
  from pluggybot.challenge import table
  model, data = _shelf_world()
  body = StubBody(model, data)
  hand = model.geom(f"{shelf.SHELF}_board_0").id     # stands in for a pad of the robot's
  body.geom_ids = np.array([hand], dtype=np.int32)
  watch = shelf.Watch(model, data, body)
  assert watch.pieces == st.PRODUCTS and set(watch.fastest_free) == set(st.PRODUCTS)
  candle = model.geom("candle_body").id
  held = [True]
  monkeypatch.setattr(table, "contact_pairs", lambda d: (
    np.array([[candle, hand]], dtype=np.int32) if held[0] else np.zeros((0, 2), dtype=np.int32)))
  data.qvel[model.jnt_dofadr[model.joint("candle_free").id] + 2] = -1.0
  watch()
  assert watch.fastest_free["candle"] == 0.0
  held[0] = False
  watch()
  assert watch.fastest_free["candle"] == pytest.approx(1.0) and watch.fastest_free["frame"] == 0.0
  m = shelf.measurements(shelf.measure(model, data), shelf.measure(model, data), watch, SLOTS,
                         started=0.0)
  assert m["candle"]["fastestFreeMS"] == pytest.approx(1.0) and m["falls"] == 0


# ---- the set-out -------------------------------------------------------------------------


def _in_s(x, y):
  origin, out, right = store.axes()
  p = np.array([x, y]) - origin
  return float(p @ right), float(p @ out)


def test_the_set_out_is_seeded_on_the_cart_and_each_product_its_own_label():
  cart_front = store.CART_BACK_GAP + 2 * st.CART_HALF[0]
  seen = set()
  for k in range(64):
    so = ct.shelf_set_out(k)
    assert so == ct.shelf_set_out(k)
    assert set(so.slots) == set(st.PRODUCTS) and set(so.slots.values()) <= set(st.LABEL_IDS)
    assert so.slots["candle"] != so.slots["frame"]
    seen |= {(p, i) for p, i in so.slots.items()}
    for p, (x, y, yaw) in so.products.items():
      (s0, d0), (ds, dd), turn = ct.SHELF_SOURCES[p]
      s, d = _in_s(x, y)
      assert abs(s - store.CART_AT - s0) <= ds + 1e-9 and abs(d - cart_front - d0) <= dd + 1e-9
      assert abs(math.remainder(yaw - store.facing_in(), 2 * math.pi)) <= turn + 1e-9
      # ...on the cart's top, clear of its edges
      assert abs(s - store.CART_AT) < st.CART_HALF[1] - 0.06
      assert store.CART_BACK_GAP + 0.06 < d < cart_front - 0.06
    c, f = (np.array(so.products[p][:2]) for p in st.PRODUCTS)
    assert np.linalg.norm(c - f) > 0.2
    # the robot faces the shelf from in front of the cart's middle
    sx, sy, heading = so.start
    ahead = ct._stance_ahead()
    s, d = _in_s(sx + ahead * math.cos(heading), sy + ahead * math.sin(heading))
    assert (s, d) == pytest.approx((store.CART_AT, cart_front + ct.CART_GAP))
    assert heading == pytest.approx(store.facing_in())
  assert len(seen) == 2 * len(st.LABEL_IDS), "some product never drew some label"


def test_a_set_outs_places_are_not_its_products_hidden_draw():
  """The cube's lesson (#525): every product's place is drawn off the
  set-out's stream, its weight off the stock's."""
  ks = range(32)
  for p in st.PRODUCTS:
    placed = [_in_s(*ct.shelf_set_out(k).products[p][:2])[0] for k in ks]
    hidden = [getattr(st.draw(k), f"{p}_kg") for k in ks]
    assert abs(np.corrcoef(placed, hidden)[0, 1]) < 0.9, p
  for k in ks:
    load = st.draw(k)
    for n, (lo, hi) in st.RANGES.items():
      assert lo <= getattr(load, n) <= hi


# ---- put away and set out ------------------------------------------------------------------


def _house_with_stock():
  spec = mujoco.MjSpec.from_file(ct.HOME_XML)
  store.attach(spec, ct.shelf_set_out(0))
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data


def test_the_stock_is_compiled_put_away_set_out_and_put_away_again():
  """As the dishware (#560): compiled under the floor, out of every cast,
  a product's weight held; an offer's draw sets it out live on the cart,
  its weights and friction drawn, and it stands; put away, it holds still
  under the floor again -- the hold counted by a redraw made while it was
  out (`dishware.set_const`, the period of `loop-k4`)."""
  from pluggybot.perception.cull import PARKED_GROUP
  model, data = _house_with_stock()
  assert store.present(model) and store.away(model, data, store.SHELF)
  for name in st.BODIES:
    b = model.body(name).id
    for g in range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b]):
      assert model.geom_group[g] == PARKED_GROUP and model.geom_contype[g] == 0, name
  act = store.activity_for(model, data, store.SHELF, 0)
  act.sense(model, data)
  assert act.flags["setOut"] is False
  k = 5
  store.set_out(model, data, store.SHELF, k)
  mujoco.mj_forward(model, data)
  load = st.draw(k)
  for p in st.PRODUCTS:
    b = model.body(p).id
    assert model.body_mass[b] == pytest.approx(getattr(load, f"{p}_kg"))
    assert model.geom_friction[model.body_geomadr[b]][0] == pytest.approx(load.friction)
    assert model.body_gravcomp[b] == 0.0
  for _ in range(500):
    mujoco.mj_step(model, data)
  so = ct.shelf_set_out(k)
  for p in st.PRODUCTS:
    at = data.xpos[model.body(p).id]
    assert np.allclose(at, [*so.products[p][:2], st.CART_TOP], atol=2e-3), (p, at)
  act.renew(model, load, so.slots)
  act.sense(model, data)
  assert act.flags["setOut"] is True and act.slots == so.slots
  m = shelf.measure(model, data)
  assert set(m["labels"]) == {str(i) for i in st.LABEL_IDS}
  assert store.label_along(so.slots["candle"]) == pytest.approx(m["labels"][str(so.slots["candle"])]["s"])
  # ...redrawn while out, then put away: its weight is held, it stays put
  store.redraw(model, data, store.SHELF, k + 1)
  store.put_away(model, data, store.SHELF)
  for _ in range(500):
    mujoco.mj_step(model, data)
  assert store.away(model, data, store.SHELF)
  act.sense(model, data)
  assert act.flags["setOut"] is False


def test_the_stock_is_not_in_the_served_house():
  """#736's stage 1 offers nothing: the served house carries no stock, so
  its geometry, its fingerprint and the scenes the site vendors are as they
  were. It comes in with the job that offers it."""
  from pluggybot.lifecycle import HUMANOID_HOME, world_config
  from pluggybot.robot import world_spec
  model = world_spec(world_config(HUMANOID_HOME)["model"]).compile()
  assert not store.present(model)
  names = {model.body(b).name for b in range(model.nbody)}
  assert not names & set(st.BODIES)


def test_the_served_houses_hold_is_counted_as_before(monkeypatch):
  """`dishware.set_const` counts the products' hold too, and in the served
  house -- no stock in it -- an offer's redraw holds the dishware's pieces
  alone at `mj_setConst`, as it did, and puts them away held."""
  from pluggybot.home import restaurant
  from pluggybot.lifecycle import HUMANOID_HOME, world_config
  from pluggybot.robot import world_spec
  model = world_spec(world_config(HUMANOID_HOME)["model"]).compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  seen, real = [], mujoco.mj_setConst

  def spy(m, d):
    seen.append({m.body(b).name for b in range(m.nbody) if m.body_gravcomp[b] > 0})
    return real(m, d)
  monkeypatch.setattr(mujoco, "mj_setConst", spy)
  restaurant.set_out(model, data, restaurant.TABLE, 3)
  assert seen == [set(dw.PIECES)] and model.ngravcomp == len(dw.PIECES)
  assert all(model.body_gravcomp[model.body(p).id] == 0.0 for p in dw.PIECES)


def test_a_cast_meets_the_cart_where_it_stands():
  """The cart moves (a mocap body), so it keeps MuJoCo's own inertia -- set
  unturned at its middle, its ray-cull sphere where its boxes are (#657)."""
  model, data = _house_with_stock()
  store.set_out(model, data, store.SHELF, 0)
  mujoco.mj_forward(model, data)
  b = model.body(st.CART).id
  assert np.allclose(data.ximat[b].reshape(3, 3), data.xmat[b].reshape(3, 3))
  x, y, _ = store.cart_place()
  geomid = np.array([-1], dtype=np.int32)
  # straight down onto the cart's top, through each corner's 2 cm in
  for sx in (-1, 1):
    for sy in (-1, 1):
      c = np.array([x, y]) + store.axes()[1] * sx * (st.CART_HALF[0] - 0.02) \
        + store.axes()[2] * sy * (st.CART_HALF[1] - 0.02)
      dist = mujoco.mj_ray(model, data, np.array([c[0], c[1], 1.5]), np.array([0, 0, -1.0]),
                           None, 1, -1, geomid)
      assert dist == pytest.approx(1.5 - st.CART_TOP, abs=1e-6)
      assert model.geom(int(geomid[0])).name == f"{st.CART}_geom"


# ---- the tags and the row -------------------------------------------------------------------


def test_the_tags_are_the_registrys_each_its_own_and_drawn():
  others = [i for ids in (tags.BLOCK_TAG_IDS, tags.MASS_TAG_IDS, tags.PLATE_TAG_IDS,
                          tags.CHEST_TAG_IDS, tags.DRAWER_TAG_IDS, tags.DOOR_TAG_IDS,
                          tags.CUBE_TAG_IDS, tags.MAT_TAG_IDS, tags.STATION_TAG_IDS,
                          tags.COUNTER_TAG_IDS, tags.STATION_2_TAG_IDS, tags.STATION_3_TAG_IDS,
                          tags.RESTAURANT_TAG_IDS, tags.SHOP_TAG_IDS, tags.PARK_TAG_IDS,
                          tags.ROOM_TAG_IDS, tags.STUDY_TAG_IDS, tags.DISH_TAG_IDS,
                          tags.PLACE_TAG_IDS) for i in ids]
  others += [i for ids in tags.BOARD_TAG_IDS.values() for i in ids]
  mine = [*tags.STOCK_TAG_IDS, *tags.LABEL_TAG_IDS]
  assert not set(mine) & set(others) and len(set(mine)) == len(mine)
  # #736's call 7: 76-78 the products (the lamp's 78 its own round's), 79-84 the labels
  assert set(mine) <= set(range(76, 85)) and 78 not in mine
  for i in mine:
    assert tags.TAG_SIZES[i] == tags.STOCK_TAG_SIZES[i]
    assert (Path(__file__).parent.parent / "models" / "tags" / f"tag{i}.png").exists(), i
  assert st.TAG_OF == dict(zip(st.PRODUCTS, tags.STOCK_TAG_IDS))
  assert tags.TAG_SIZES[tags.LABEL_TAG_IDS[0]] == 0.040                 # call 3: 40 mm


def test_the_products_are_shaped_for_the_gripper_they_meet():
  from pluggybot.humanoid import parts
  for across in (2 * st.CANDLE_R, 2 * st.PANEL[1]):
    assert across < parts.GRIPPER.stroke - 0.010


def test_the_shelf_pays_nothing_and_is_shown_to_no_mind():
  row = json.loads((SRC / "economy" / "challenges.json").read_text())["tasks"]["shelf"]
  assert row["base"] == 0 and row.get("bonus", 0) == 0 and row["shown"] is False
  assert "shelf" in scoring.EVALUATORS and "shelf" in scoring.SAMPLERS
  assert "shelf" not in scoring.default_table().offered
  # ...its words, the criteria's numbers
  assert "4 cm" in row["detail"] and "10 cm" in row["detail"]


def test_our_reference_is_told_each_set_outs_labels_as_an_offer_would():
  """Our procedure (`hands_check.SHELF`) is one source with the job's labels
  filled in per set-out (`hands_check.source`), as an offer states them,
  and compiles in the hands' vocabulary within the source's cap."""
  from pluggybot.evaluation import hands_check as hc
  from pluggybot.procedure import lang
  for k in range(8):
    so = ct.shelf_set_out(k)
    src = hc.source("shelf", k)
    assert f"  c = {so.slots['candle']}\n" in src and f"  f = {so.slots['frame']}\n" in src
    assert len(src) <= lang.MAX_SOURCE_CHARS
    assert hc.compiled("shelf", k).name == "stock_shelf"
