"""The shop's shelf stocked as a JOB (issue #736's stage 1, not offered): the
stock compiled put away into a house that has it (`home.store`; no served
house does yet), an offer of `stock_shelf` setting it out at its draw, its
labels the offer's words, watched from the claim and graded after `done`
against those labels -- each rule pinned on a stub whose world is the shop's
shelf alone (`StubBody`), nothing flown. Our reference flown as the job is
`scripts/solve.py --feature shelf`'s.
"""

import math
import re

import mujoco
import numpy as np
import pytest

from pluggybot.activity import dishware as dw
from pluggybot.activity import stock as st
from pluggybot.activity.base import ActivitySet
from pluggybot.body import STUB_WORLD, StubBody
from pluggybot.economy.ledger import Ledger
from pluggybot.economy.scoring import default_table
from pluggybot.economy.tasks import KINDS, MAX_DESCRIPTION, Task, TaskBoard
from pluggybot.evaluation import counter as ct
from pluggybot.home import jobs, store
from pluggybot.home import world as hw
from pluggybot.lifecycle import (CHECK_S, HUMANOID_HOME, HubLifecycle, task_producer,
                                 world_config)
from pluggybot.mind import overseer as ov


class _Mind:
  """A mind as the claim reads one: a library, and nothing that decides."""
  event_map = None
  pending = None
  interrupt_pending = None
  can_escalate = False
  spend = None
  library = object()
  decisions: list = []
  model, backend = "none", "stub"
  reasoning_effort = None


def _shelf_world():
  """The stub's floor with the shop's shelf on it as the house draws it and
  the stock put away (`store.attach`), compiled."""
  spec = mujoco.MjSpec.from_string(STUB_WORLD)
  spec.texturedir = "models/"
  name, cx, cy, length, depth, yaw, double = next(s for s in hw.SHELVES if s[0] == store.SHELF)
  body = spec.worldbody.add_body(name=name, pos=[cx, cy, 0.0],
                                 quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
  for s, c, hh in hw.shelf_geoms(length, depth, double):
    body.add_geom(name=f"{name}_{s}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(c), size=list(hh))
  store.attach(spec, ct.stock_set_out(0))
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data


def _life(tmp_path):
  model, data = _shelf_world()
  body = StubBody(model, data, grid_bounds=(19.0, -25.0, 25.0, -17.0))
  life = HubLifecycle(model, data, realtime=False, world=HUMANOID_HOME, body=body,
                      battery_wh=world_config(HUMANOID_HOME)["hosting_battery_wh"],
                      tasks=TaskBoard(path=str(tmp_path / "tasks.json")),
                      ledger=Ledger(path=str(tmp_path / "ledger.json")), overseer=_Mind())
  acts = ActivitySet(store.activities(model, data))
  body.step_hooks.append(acts.step_hook(model, data))
  life.activities = acts
  life.tasks.seq = max(life.tasks.seq, 5)
  return life


def _offer(life):
  """The shelf's job offered as the cadence offers it (`TaskProducer._build`:
  its draw's labels filled in by the producer, `home.jobs.words`)."""
  maker = task_producer(life.tasks, HUMANOID_HOME, procedures=True)
  made, secret = maker._build("stock_shelf", store.SHELF)
  return life.tasks.offer("stock_shelf", store.SHELF, params=made, secret=secret,
                          ttl=3000.0, t=float(life.data.time))


def _at(life, name: str) -> np.ndarray:
  return dw._placed_at(life.model, life.data, life.model.body(name).id)


def _grade(life, task_id: str) -> dict:
  life._done(ov.Decision(action="idle", reason="", done=task_id))
  life.body.run(life._grade_routine())
  return dict(life.tasks.get(task_id).verdict or {})


def _stock(life, slots: dict) -> None:
  """Each product at rest over the label `slots` names, its front 5 cm
  behind the board's edge, facing the aisle."""
  front = {"candle": st.CANDLE_R, "frame": st.PANEL[0]}
  for p, label in slots.items():
    x, y = store.to_world(store.label_along(label), -0.05 - front[p])
    dw._pose(life.model, life.data, life.model.body(p).id,
             (x, y, store.board_top(st.BOARD_OF[label])), dw._yaw_quat(store.facing_in()))
  mujoco.mj_forward(life.model, life.data)


def test_an_offer_sets_the_stock_out_at_its_draw_and_a_settled_job_puts_it_away(tmp_path):
  """Its sentence names its draw's labels as the producer builds the offer
  (round 1 of the review: filled by `job_routine` alone, a cadence's offer
  would have read "over label , the frame over label ,")."""
  life = _life(tmp_path)
  assert store.away(life.model, life.data, store.SHELF), "compiled put away"
  task = _offer(life)
  k = task.secret["draw"]
  so, load = ct.stock_set_out(k), st.draw(k)
  # ...its sentence names its draw's labels, says when, within the cap
  assert f"label {so.slots['candle']}" in task.description
  assert f"label {so.slots['frame']}" in task.description
  assert "These were set out at t=" in task.description
  for p, (x, y, _) in so.products.items():
    assert _at(life, p)[:2] == pytest.approx((x, y)), p
    assert life.model.body_mass[life.model.body(p).id] == pytest.approx(getattr(load, f"{p}_kg"))
  act = life._target_activity(store.SHELF)
  assert act.slots == so.slots, "the activity knows its draw's labels"
  assert life._claim_task(task.id)
  life.tasks.resolve(task.id, _verdict(), t=float(life.data.time))
  life.data.time = max(float(life.data.time), life._next_task_check) + CHECK_S
  life._task_step()
  assert store.away(life.model, life.data, store.SHELF), "settled, put away"


def _verdict():
  from pluggybot.economy.scoring import evaluate
  return evaluate("stock_shelf", {})


def test_the_shelf_stocked_over_its_draws_labels_passes_and_pays_nothing(tmp_path):
  """The grade on the seam reads the products against the labels its OWN
  draw names (the activity's, `_hands_measured`'s `slots`): stocked over
  them it passes, over the two swapped it does not."""
  life = _life(tmp_path)
  task = _offer(life)
  slots = ct.stock_set_out(task.secret["draw"]).slots
  assert life._claim_task(task.id)
  _stock(life, slots)
  verdict = _grade(life, task.id)
  assert verdict["ok"], verdict["reason"]
  assert default_table()["stock_shelf"].base == 0
  life = _life(tmp_path / "again")
  task = _offer(life)
  assert life._claim_task(task.id)
  _stock(life, {"candle": slots["frame"], "frame": slots["candle"]})
  verdict = _grade(life, task.id)
  assert not verdict["ok"] and f"label {slots['candle']}" in verdict["reason"], verdict["reason"]


def test_the_kind_is_offered_nowhere_and_its_sentence_fits_every_draw():
  """No world's cadence names `stock_shelf`, and the served house has no
  stock to set out; its sentence, labels filled, fits a description's cap
  at a 7-digit clock for any draw."""
  from pluggybot.economy.cadence import default_cadence
  for world in ("", HUMANOID_HOME):
    assert "stock_shelf" not in default_cadence(world).kinds
  assert jobs.area_of(store.SHELF) is store and KINDS["stock_shelf"].target_kind == "shelf"
  for k in range(64):
    t = Task.create("stock_shelf", store.SHELF, "t_1", params=store.labels(k), t=9_999_999.0)
    assert len(t.description) < MAX_DESCRIPTION and t.description.endswith("procedure.")
  assert dict(store.drawn(store.SHELF, 3).slots) == store.labels(3) == ct.stock_set_out(3).slots


def test_the_offer_says_where_by_the_shelfs_own_area_and_never_a_position():
  """The shelf's offer carries its own area's terms, as every hand job's
  does (#419): the mall's address and the shop shelf's directions
  (`places.json`), offered and held, on the wire too, its sentence no
  position -- and the sentence and the directions name the one shelf they
  mean, the back shelf (the store has a middle one too)."""
  from pluggybot.home import places as addresses
  board = TaskBoard(path=None)
  params, secret = task_producer(board, HUMANOID_HOME, procedures=True)._build(
    "stock_shelf", store.SHELF)
  job = board.offer("stock_shelf", store.SHELF, params=params, secret=secret, t=1.0)
  area = addresses.load()["areas"][store.SHELF]
  assert area["house"] == "mall"
  for shown in (job.as_context(1.0), job.as_held(1.0), job.as_dict()["params"]):
    assert shown["address"] == addresses.address("mall") == addresses.area(store.SHELF)["address"]
    assert shown["directions"] == area["directions"]
  assert "the store's back shelf" in job.description and "the back shelf's" in area["directions"]
  assert not re.search(r"-?\d+\.\d+\s*,\s*-?\d+\.\d+", job.description), job.description


def test_the_shelfs_directions_say_where_an_offer_sets_the_stock_out():
  """#723's rule at the shelf: its directions say where an offer's things
  are -- the labels on the front edges of the back shelf's second and third
  boards from the floor, from left to right as you face them; the candle
  and the frame on a cart standing at the shelf beside the labels, on their
  left; the candle's tag on its top and the frame's on its face -- true at
  every draw by the set-out's own code, so the words move with it."""
  from pluggybot.home import places as addresses
  words = addresses.load()["areas"][store.SHELF]["directions"]
  assert ("the back shelf's second and third boards from the floor mark its slots: tags 79, 80 "
          "and 81 on the second board and 82, 83 and 84 on the third, from left to right as you "
          "face them") in words
  boards = sorted(range(len(hw.SHELF_BOARDS_Z)), key=lambda b: hw.SHELF_BOARDS_Z[b])
  for row, board in ((st.LABEL_IDS[:3], boards[1]), (st.LABEL_IDS[3:], boards[2])):
    assert row == ((79, 80, 81) if board == boards[1] else (82, 83, 84))
    assert {st.BOARD_OF[i] for i in row} == {board}
    along = [store.label_along(i) for i in row]
    assert along == sorted(along) and len(set(along)) == 3, "left to right as you face them"
  assert ("set out on a cart standing at the back shelf, beside the labels on their left as you "
          "face them") in words
  origin, out, right = store.axes()
  assert store.CART_AT + st.CART_HALF[1] < min(store.label_along(i) for i in st.LABEL_IDS)
  for k in range(64):
    for p, (x, y, _) in ct.stock_set_out(k).products.items():
      at = np.array([x, y]) - origin
      s, d = float(at @ right), float(at @ out)
      assert abs(s - store.CART_AT) < st.CART_HALF[1], (k, p, "along the cart")
      assert store.CART_BACK_GAP < d < store.CART_BACK_GAP + 2 * st.CART_HALF[0], (k, p, "on it")
  assert "The candle carries tag 76 on its top and the frame tag 77 on its face" in words
  assert st.TAG_OF == {"candle": 76, "frame": 77}
  model, _ = _shelf_world()
  candle, frame = model.geom("candle_tag"), model.geom("frame_tag")
  assert candle.pos[2] > st.CANDLE_H and abs(candle.pos[0]) + abs(candle.pos[1]) == 0, "its top"
  assert frame.pos[0] < -st.PANEL[0] and abs(frame.pos[2] - st.PANEL[2]) < 1e-9, "its face"


def test_the_labels_are_places_only_where_the_stock_is():
  """The body's place memory takes the labels where a house has the stock
  (`place_tags`), and the served house's places are as they were. ⚠ A
  procedure's `find` still takes none of them: the library checks a source
  against its world's facts (`world_facts`, the served house's places), so
  the shelf job's walk finds the shop's wall tag (`solutions.STOCK`)."""
  from pluggybot.humanoid.places import place_tags
  from pluggybot.perception.tags import LABEL_TAG_IDS
  from pluggybot.robot import world_spec
  served = world_spec(world_config(HUMANOID_HOME)["model"]).compile()
  assert not set(LABEL_TAG_IDS) & set(place_tags(served)[0])
  model, _ = _shelf_world()
  assert set(LABEL_TAG_IDS) <= set(place_tags(model)[0])


def test_the_stock_put_into_a_house_with_robots_is_compiled_put_away():
  """`solve.py --feature shelf`'s house: home_spec's with its robot in it,
  the stock put in by its own call. Such a spec finds no body added after
  the robot by its name, and parked by name the first flight raised
  (`dishware.park_body`, by the bodies themselves)."""
  from pluggybot.robot import world_spec
  spec = world_spec(world_config(HUMANOID_HOME)["model"])
  store.attach(spec, ct.stock_set_out(0))
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  assert store.present(model) and store.away(model, data, store.SHELF)


def test_the_served_house_offers_no_shelf_job():
  """Review round 1: with the store an area of `home.jobs`, the served
  house's config named `shop_shelf` a target it has no stock for. Not
  served (`store.SERVED`), it is no world's target until the release that
  puts the stock in."""
  from pluggybot.lifecycle import world_targets
  assert "shelf" not in world_targets(HUMANOID_HOME, procedures=True)
  assert store.SHELF not in world_config(HUMANOID_HOME)["jobs"]
