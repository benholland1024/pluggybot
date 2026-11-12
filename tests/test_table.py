"""A table set (issue #560's first task, easy mode): what it stands on,
each pinned without a flight (docs/Testing.md) -- its flights are
`scripts/hands_check.py --job table`'s.

  1. the grader (`challenge/table.py`) passes only what its criteria say,
     each criterion failing alone, judged unrounded, a fall first;
  2. the place's frame is the card's, and a piece is measured in it: at its
     spot, turned, raised;
  3. a piece the robot holds is never dropped, however fast it is carried
     down: only a free piece's fall is one;
  4. the set-out is seeded, its pieces inside their ranges and apart, their
     hidden draw a stream of its own;
  5. the dishware's activity senses a piece lifted, and keeps nothing of
     its load in a flag;
  6. the tags are the registry's, and the row pays nothing, shown to no mind.
"""

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from pluggybot.activity import dishware as dw
from pluggybot.challenge import table
from pluggybot.economy import scoring
from pluggybot.evaluation import counter as ct
from pluggybot.perception import tags

SRC = Path(__file__).parent.parent / "src" / "pluggybot"
BARE = """<mujoco model="table"><option integrator="implicitfast"/>
  <worldbody><geom name="floor" type="plane" size="3 3 0.1"/></worldbody></mujoco>"""


# ---- the grader: each criterion fails alone -----------------------------------------


def _laid(**kw):
  piece = {"liftedM": 0.05, "fastestFreeMS": 0.1, "offAtCallM": 0.01, "offM": 0.01,
           "aboveM": 0.0, "tiltDeg": 1.0}
  m = {"robotTiltDeg": 5.0, "falls": 0, "touchedDuringHold": [], "holdS": table.HOLD_S,
       "elapsedS": 90.0, "limitS": 300.0,
       **{p: dict(piece) for p in dw.PIECES}}
  m["fork"]["turnDeg"] = 3.0
  for k, v in kw.items():
    if isinstance(v, dict):
      m[k] = {**m[k], **v}
    else:
      m[k] = v
  return m


def test_the_place_passes_laid_let_go_in_time_and_on_nothing_less():
  ok, metrics, why = table.eval_table(_laid())
  assert ok and why.startswith("the place laid in 90 s"), why
  for p in dw.PIECES:
    ok, _, why = table.eval_table(_laid(**{p: {"offM": table.PLACE_M + 0.001}}))
    assert not ok and why.startswith(f"the {p} ended 3 cm from its place"), why
    ok, _, why = table.eval_table(_laid(**{p: {"aboveM": 0.03}}))
    assert not ok and "3 cm over the table" in why, why
    ok, _, why = table.eval_table(_laid(**{p: {"tiltDeg": math.degrees(table.UPRIGHT_RAD) + 0.5}}))
    assert not ok and why.startswith(f"the {p} rests at its place tipped"), why
    ok, _, why = table.eval_table(_laid(**{p: {"fastestFreeMS": table.DROP_M_S + 0.01}}))
    assert not ok and why.startswith(f"the {p} came down at 0.5 m/s with nothing holding it"), why
  assert "upright" in table.eval_table(_laid(cup={"tiltDeg": 30.0}))[2]
  assert "flat" in table.eval_table(_laid(plate={"tiltDeg": 30.0}))[2]
  ok, _, why = table.eval_table(_laid(fork={"turnDeg": math.degrees(table.FORK_RAD) + 0.5}))
  assert not ok and why.startswith("the fork lies 15 deg off the place's line"), why
  # laid backwards -- its head toward the diner -- is no fork laid
  assert not table.eval_table(_laid(fork={"turnDeg": 180.0}))[0]
  ok, _, why = table.eval_table(_laid(touchedDuringHold=["left_jaw_a_pad"]))
  assert not ok and "still touched it after the call" in why
  ok, _, why = table.eval_table(_laid(elapsedS=301.0))
  assert not ok and why.startswith("called 301 s after the start")
  ok, _, why = table.eval_table(_laid(holdS=1.0))
  assert not ok and why.startswith("measured 1.0 s after the call")


def test_the_place_is_judged_as_measured_shown_rounded_and_a_fall_first():
  m = _laid(plate={"offM": 0.0254})
  ok, metrics, _ = table.eval_table(m)
  assert not ok and metrics["plate"]["offM"] == 0.025
  ok, metrics, why = table.eval_table(_laid(falls=1))
  assert not ok and why.startswith("you fell, and a fall fails the try -- the place laid")
  assert table.eval_table(_laid(falls=None))[::2] == (False, "your falls were never counted")
  # a missing measurement is not a passing one
  for p in dw.PIECES:
    assert table.eval_table(_laid(**{p: {"offM": None}}))[::2] == (
      False, "the place was never measured"), p
  assert not table.eval_table(_laid(fork={"turnDeg": None}))[0]
  # ...and a tilted body is judged by nothing
  assert table.eval_table(_laid(robotTiltDeg=80.0))[0]


# ---- the place's frame ------------------------------------------------------------------


def _place_world(card=(0.6, 0.2, -math.pi / 2), at=None, z=0.0):
  """A card on the floor and the pieces at `at` (piece -> (x, y, yaw) in the
  world), or at their spots off the card -- no robot."""
  spec = mujoco.MjSpec.from_string(BARE)
  dw.attach_card(spec, card[:2], card[2], z, tags=False)
  c, s = math.cos(card[2]), math.sin(card[2])
  toward, left = np.array([c, s]), np.array([s, -c])
  load = dw.Load()
  for p in dw.PIECES:
    if at and p in at:
      x, y, yaw = at[p]
    else:
      a, b = table.SPOTS[p]
      x, y = np.array(card[:2]) + a * toward + b * left
      yaw = math.atan2(-toward[1], -toward[0])
    dw.ATTACH[p](spec, (x, y), yaw, z, getattr(load, f"{p}_kg"), load.friction, tags=False)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data


def test_the_place_is_the_cards_and_a_piece_is_measured_in_it():
  """The spots are off the card's TRUE pose, toward the diner and to the
  diner's left: pieces put at them read nothing off, whichever way the card
  faces; a fork laid head toward the diner reads half a turn."""
  for yaw in (-math.pi / 2, 0.4, 2.5):
    model, data = _place_world(card=(0.6, 0.2, yaw))
    got = table.measure(model, data)
    for p in dw.PIECES:
      assert got[p]["offM"] == pytest.approx(0.0, abs=1e-9), (yaw, p)
      assert got[p]["aboveM"] == pytest.approx(0.0, abs=1e-6), (yaw, p)
      assert got[p]["tiltDeg"] == pytest.approx(0.0, abs=1e-6), (yaw, p)
    assert got["fork"]["turnDeg"] == pytest.approx(0.0, abs=1e-6), yaw
  # the diner's left: the card faces -y (the diner south of it), so the
  # diner faces +y and their left is -x
  origin, toward, left, top = table.place_frame(*_place_world()[:2])
  assert np.allclose(toward, [0, -1, 0]) and np.allclose(left, [-1, 0, 0]) and top == 0.0
  model, data = _place_world()
  fork = data.xpos[model.body("fork").id].copy()
  model, data = _place_world(at={"fork": (fork[0], fork[1], -math.pi / 2)})
  assert abs(table.measure(model, data)["fork"]["turnDeg"]) == pytest.approx(180.0, abs=1e-6)


def test_a_piece_turned_over_is_no_piece_laid():
  q = [math.cos(math.pi / 2), math.sin(math.pi / 2), 0.0, 0.0]          # upside down
  assert math.degrees(table.resting_tilt(q)) == pytest.approx(180.0)
  q = [math.cos(math.radians(10)), math.sin(math.radians(10)), 0.0, 0.0]
  assert math.degrees(table.resting_tilt(q)) == pytest.approx(20.0)
  assert table.resting_tilt([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)


# ---- a held piece is never dropped -------------------------------------------------------


def test_a_piece_the_robot_holds_is_carried_down_never_dropped(monkeypatch):
  """The cube's grader read a held cube swinging down at up to 1 m/s with
  the carry as a drop (#560, its check's set-outs 3 and 5): here only a
  piece nothing of the robot touches can be dropped -- the cup coming down
  at 1 m/s reads nothing while a geom of the robot's touches it, and the
  same speed with nothing touching it is a fall."""
  from pluggybot.body import StubBody
  model, data = _place_world()
  body = StubBody(model, data)
  hand = model.geom("place_card").id               # stands in for a pad of the robot's
  body.geom_ids = np.array([hand], dtype=np.int32)
  watch = table.Watch(model, data, body)
  cup = model.geom("cup").id
  held = [True]
  monkeypatch.setattr(table, "contact_pairs", lambda d: (
    np.array([[cup, hand]], dtype=np.int32) if held[0] else np.zeros((0, 2), dtype=np.int32)))
  data.qvel[model.jnt_dofadr[model.joint("cup_free").id] + 2] = -1.0
  watch()
  assert watch.fastest_free["cup"] == 0.0
  held[0] = False
  watch()
  assert watch.fastest_free["cup"] == pytest.approx(1.0)
  assert watch.fastest_free["plate"] == 0.0 and watch.fastest_free["fork"] == 0.0


# ---- the set-out ---------------------------------------------------------------------------


def _in_t(x, y):
  """A world point in the place's frame T (`counter.place_to_world`'s inverse)."""
  tx, ty = ct._dining_table()
  return y - (ty - ct._table_half()), tx - x


def test_the_set_out_is_seeded_inside_its_ranges_and_its_pieces_apart():
  for k in range(64):
    so = ct.table_set_out(k)
    assert so == ct.table_set_out(k)
    for p, (x, y, yaw) in so.pieces.items():
      (sx, sy), (dx, dy), turn = ct.TABLE_SOURCES[p]
      px, py = _in_t(x, y)
      assert abs(px - sx) <= dx + 1e-9 and abs(py - sy) <= dy + 1e-9, (k, p)
      assert abs(math.remainder(yaw - math.pi / 2, 2 * math.pi)) <= turn + 1e-9, (k, p)
    plate, cup, fork = (np.array(so.pieces[p][:2]) for p in ("plate", "cup", "fork"))
    # the fork's handle and head lie along its turn, in from its middle
    head = fork + (dw.HANDLE[0] + 2 * dw.HEAD[0]) * np.array(
      [math.cos(so.pieces["fork"][2]), math.sin(so.pieces["fork"][2])])
    for q in np.linspace(fork, head, 12):
      assert np.linalg.norm(q - plate) > dw.PLATE_R + dw.HANDLE[1], k
    assert np.linalg.norm(cup - plate) > dw.PLATE_R + dw.CUP_R, k
  # the robot stands at the place facing in, its soles `TABLE_GAP` out
  so = ct.table_set_out(0)
  sx = so.start[0]
  sy = so.start[1] + ct._stance_ahead()
  assert _in_t(sx, sy) == pytest.approx((-ct.TABLE_GAP, 0.0))
  assert so.start[2] == pytest.approx(math.pi / 2)
  assert _in_t(*so.card[:2]) == pytest.approx((ct.CARD_IN, 0.0))


def test_a_set_outs_places_are_not_its_pieces_hidden_draw():
  """The cube's lesson (#525): off one seed, the furthest cube was always
  the heaviest. Every piece's place is drawn off the set-out's stream, its
  weight off the dishware's."""
  ks = range(32)
  for p in dw.PIECES:
    placed = [_in_t(*ct.table_set_out(k).pieces[p][:2])[1] for k in ks]
    hidden = [getattr(dw.draw(k), f"{p}_kg") for k in ks]
    assert abs(np.corrcoef(placed, hidden)[0, 1]) < 0.9, p


def test_the_set_outs_world_is_its_table_on_legs_its_chair_away_and_its_pieces_resting():
  from pluggybot.home import world as hw
  model = ct.model_of(ct.table_set_out(0))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  names = {model.geom(g).name for g in range(model.ngeom)}
  assert f"dining_table_{ct.TABLE_I}_leg0" in names
  assert f"dining_table_{ct.TABLE_I}_chair_1_geom" not in names
  assert f"dining_table_{ct.TABLE_I}_chair_0_geom" in names
  top = data.geom_xpos[model.geom(f"dining_table_{ct.TABLE_I}_geom").id][2]
  assert top + hw.TABLE_TOP_T / 2 == pytest.approx(ct.TABLE_H)
  for _ in range(250):
    mujoco.mj_step(model, data)
  for p in dw.PIECES:
    assert float(np.abs(data.qvel[model.jnt_dofadr[model.joint(f"{p}_free").id]:][:6]).max()) < 0.02, p


# ---- the dishware's activity and its tags --------------------------------------------------


def test_a_piece_reads_lifted_with_hysteresis_and_no_flag_carries_its_load():
  model, data = _place_world()
  act = dw.Dishware(model, data, dw.draw(3), 0.0)
  cup = model.jnt_qposadr[model.joint("cup_free").id]
  for z, want in ((0.0, False), (dw.LIFT_ON + 0.001, True), (dw.LIFT_ON * 0.6, True),
                  (dw.LIFT_ON * 0.4, False)):
    data.qpos[cup + 2] = z
    mujoco.mj_forward(model, data)
    act.sense(model, data)
    assert act.flags["cupLifted"] is want, z
  assert set(act.flags) == {"setOut", *(f"{p}Lifted" for p in dw.PIECES)}
  data.qpos[cup + 2] = 0.05
  mujoco.mj_forward(model, data)
  act.sense(model, data)
  again = dw.Dishware(model, data, dw.draw(3), 0.0)
  again.restore_kept(json.loads(json.dumps(act.kept_state())))
  assert again.lifted["cup"].value is True and again.lifted["plate"].value is False


def test_the_dishware_is_shaped_for_the_gripper_it_meets():
  """Each piece fits the Dex1-1's 90 mm, and jaws stopped on it read clear
  of closed: 1.5 `HELD_GAP_M` is a margin chosen, never a limit found (a 6
  mm wall read held at all 32 of our reference's grips, #646)."""
  from pluggybot.humanoid import arms, parts
  assert dw.PLATE_WALL_T >= 1.5 * arms.HELD_GAP_M
  assert 2 * dw.HANDLE[1] >= 1.5 * arms.HELD_GAP_M
  for across in (2 * dw.CUP_R, dw.PLATE_WALL_T, 2 * dw.HANDLE[1]):
    assert across < parts.GRIPPER.stroke - 0.015


def test_the_tags_are_the_registrys_each_its_own_and_drawn():
  others = [i for ids in (tags.BLOCK_TAG_IDS, tags.MASS_TAG_IDS, tags.PLATE_TAG_IDS,
                          tags.CHEST_TAG_IDS, tags.DRAWER_TAG_IDS, tags.DOOR_TAG_IDS,
                          tags.CUBE_TAG_IDS, tags.MAT_TAG_IDS, tags.STATION_TAG_IDS,
                          tags.COUNTER_TAG_IDS, tags.STATION_2_TAG_IDS, tags.STATION_3_TAG_IDS,
                          tags.RESTAURANT_TAG_IDS, tags.SHOP_TAG_IDS, tags.PARK_TAG_IDS,
                          tags.ROOM_TAG_IDS, tags.STUDY_TAG_IDS) for i in ids]
  others += [i for ids in tags.BOARD_TAG_IDS.values() for i in ids]
  mine = [*tags.DISH_TAG_IDS, *tags.PLACE_TAG_IDS]
  assert not set(mine) & set(others) and len(set(mine)) == len(mine)
  for i in mine:
    assert tags.TAG_SIZES[i] == tags.DISH_TAG_SIZES[i]
    assert (Path(__file__).parent.parent / "models" / "tags" / f"tag{i}.png").exists(), i
  assert dw.TAG_OF == dict(zip(dw.PIECES, tags.DISH_TAG_IDS))


def test_the_place_pays_nothing_and_is_shown_to_no_mind():
  row = json.loads((SRC / "economy" / "challenges.json").read_text())["tasks"]["table"]
  assert row["base"] == 0 and row.get("bonus", 0) == 0 and row["shown"] is False
  assert "table" in scoring.EVALUATORS and "table" in scoring.SAMPLERS
  assert "table" not in scoring.default_table().offered


def test_the_practice_job_states_the_place_and_never_where_its_dishware_waits():
  from pluggybot.evaluation import attempts as at
  from pluggybot.evaluation import practice as ep
  text = ep.job_for(at.Task(chest=0, grader="table"))
  for tag in (*tags.DISH_TAG_IDS, *tags.PLACE_TAG_IDS):
    assert f"tag {tag}" in text, tag
  assert f"{100 * table.SPOTS['plate'][0]:.0f} cm from the card" in text
  assert f"{at.JOB_LIMIT_S['table']:.0f} s at most" in text
  assert ep.THINGS["table"] == "table"


def test_a_tables_reading_says_what_it_flatters():
  """A reading says what it flatters (`attempts.FLATTERS`): the table's
  its own -- its legs and its chair away -- never the lid's knob."""
  from pluggybot.evaluation import attempts as at
  said = at.flatters("table")
  assert any("top on legs" in f for f in said) and any("chair" in f for f in said)
  assert not any("knob" in f for f in said)
  assert at.WATCH_FLATTERS in said
