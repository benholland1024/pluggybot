"""The restaurant's first job (issue #560, the table set served): a place laid
at the dining room's table 0 (`home.restaurant`), its dishware put away until
an offer sets it out, watched from the claim and graded after `done` -- each
rule pinned on a stub whose world is the table alone (`StubBody`), nothing
flown. The reference flown from the tour is `scripts/solve.py --feature
table`'s.
"""

import math
import re

import mujoco
import numpy as np
import pytest

from pluggybot.activity import dishware as dw
from pluggybot.activity.base import ActivitySet
from pluggybot.body import STUB_WORLD, StubBody
from pluggybot.challenge import table
from pluggybot.economy.ledger import Ledger
from pluggybot.economy.scoring import default_table
from pluggybot.economy.tasks import HAND_MINUTES, KINDS, TABLE_MINUTES, TaskBoard
from pluggybot.evaluation import counter as ct
from pluggybot.home import restaurant
from pluggybot.home import world as hw
from pluggybot.lifecycle import (CHECK_S, HUMANOID_HOME, HubLifecycle, task_producer,
                                 world_config)
from pluggybot.mind import overseer as ov
from pluggybot.perception.cull import CAST_GROUPS, PARKED_GROUP, UNCULLED_GROUPS


class _Mind:
  """A mind as the claim reads one: a library, so a procedure job may be
  taken, and nothing that decides."""
  event_map = None
  pending = None
  interrupt_pending = None
  can_escalate = False
  spend = None
  library = object()
  decisions: list = []
  model, backend = "none", "stub"
  reasoning_effort = None


def _table_world(draw: int = 0):
  """The stub's floor with the house's own dining table 0 on it and the
  place's dishware put away (`restaurant.attach`), compiled."""
  spec = mujoco.MjSpec.from_string(STUB_WORLD)
  spec.texturedir = "models/"
  tx, ty = hw.DINING_TABLES[hw.SET_TABLE]
  hx, hy, hz = hw.DINING_TABLE_HALF
  body = spec.worldbody.add_body(name=restaurant.TABLE, pos=[tx, ty, 0.0])
  for suffix, c, h in hw.table_geoms(hx, hy, 2 * hz):
    body.add_geom(name=f"{restaurant.TABLE}_{suffix}", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=list(h), pos=list(c))
  restaurant.attach(spec, draw)
  model = spec.compile()
  return model, mujoco.MjData(model)


def _life(tmp_path, rebase: bool = True):
  """A humanoid world's lifecycle on a stub at the table: its board, its
  ledger, a mind, the place's activity on its seam."""
  model, data = _table_world()
  body = StubBody(model, data, grid_bounds=(15.0, -16.0, 26.0, -8.0))
  # ...on the served pack, which funds the table's job (`fundable_wh`)
  life = HubLifecycle(model, data, realtime=False, world=HUMANOID_HOME, body=body,
                      battery_wh=world_config(HUMANOID_HOME)["hosting_battery_wh"],
                      tasks=TaskBoard(path=str(tmp_path / "tasks.json"), rebase=rebase),
                      ledger=Ledger(path=str(tmp_path / "ledger.json")), overseer=_Mind())
  acts = ActivitySet(restaurant.activities(model, data))
  body.step_hooks.append(acts.step_hook(model, data))
  life.activities = acts
  # ...its offers drawing past the world's compiled draw (0)
  life.tasks.seq = max(life.tasks.seq, 5)
  return life


def _offer(life, **params):
  """The table's job offered as the cadence offers it."""
  maker = task_producer(life.tasks, HUMANOID_HOME, procedures=True)
  made, secret = maker._build("set_table", restaurant.TABLE)
  return life.tasks.offer("set_table", restaurant.TABLE, params={**made, **params},
                          secret=secret, ttl=3000.0, t=float(life.data.time))


def _sweep(life) -> None:
  """The board's next sweep on the seam (`_task_step`)."""
  life.data.time = max(float(life.data.time), life._next_task_check) + CHECK_S
  life._task_step()
  mujoco.mj_forward(life.model, life.data)


def _at(life, name: str) -> np.ndarray:
  return dw._placed_at(life.model, life.data, life.model.body(name).id)


def _out(life) -> bool:
  return not restaurant.away(life.model, life.data, restaurant.TABLE)


def _grade(life, task_id: str) -> dict:
  life._done(ov.Decision(action="idle", reason="", done=task_id))
  life.body.run(life._grade_routine())
  return dict(life.tasks.get(task_id).verdict or {})


def _lay(life) -> None:
  """Each piece at rest on its spot, as a place laid is (`table.SPOTS`)."""
  mujoco.mj_forward(life.model, life.data)
  origin, toward, left, top = table.place_frame(life.model, life.data)
  for p in dw.PIECES:
    a, b = table.SPOTS[p]
    at = origin + a * toward + b * left
    yaw = math.atan2(-toward[1], -toward[0]) if p == "fork" else 0.0
    dw._pose(life.model, life.data, life.model.body(p).id, (at[0], at[1], top), dw._yaw_quat(yaw))
  mujoco.mj_forward(life.model, life.data)


# ---- set out, put away ---------------------------------------------------------------------


def test_an_offer_sets_the_place_out_and_a_settled_job_puts_it_away(tmp_path):
  """The dishware waits put away -- a free body costs the physics and a geom
  every ray -- and an offer sets it out where its draw lays it, its hidden
  weights the draw's; held, it stays out; settled, the next sweep puts it
  away again."""
  life = _life(tmp_path)
  assert not _out(life), "compiled put away"
  task = _offer(life)
  assert "The pieces were set out at t=" in task.description, "says when (#659)"
  k = task.secret["draw"]
  so, load = ct.table_set_out(k), dw.draw(k)
  for p, (x, y, _) in so.pieces.items():
    assert _at(life, p)[:2] == pytest.approx((x, y)), p
    b = life.model.body(p).id
    assert life.model.body_mass[b] == pytest.approx(getattr(load, f"{p}_kg")), p
    assert life.model.geom_group[life.model.body_geomadr[b]] == 0
  assert _at(life, "place_card")[:2] == pytest.approx(so.card[:2])
  act = life._target_activity(restaurant.TABLE)
  assert act.load == load
  life.body.run(life.body.hold_routine(0.1))
  assert act.flags["setOut"] is True
  assert life._claim_task(task.id)
  _sweep(life)
  assert _out(life), "a job holds it"
  life.tasks.resolve(task.id, table_verdict(), t=float(life.data.time))
  _sweep(life)
  assert not _out(life), "settled, put away"
  life.body.run(life.body.hold_routine(0.1))
  assert act.flags["setOut"] is False


def table_verdict():
  from pluggybot.economy.scoring import evaluate
  return evaluate("set_table", {})


def test_a_lapsed_offer_puts_the_place_away(tmp_path):
  life = _life(tmp_path)
  _offer(life)
  assert _out(life)
  life.tasks.expire_due(1e9)
  _sweep(life)
  assert not _out(life)


def test_nothing_is_put_away_from_under_a_hand(tmp_path, monkeypatch):
  """A robot still holding a piece when its job is settled (out of time, a
  failed grade) keeps it: the place waits until it lets go."""
  life = _life(tmp_path)
  task = _offer(life)
  life.tasks.resolve(task.id, table_verdict(), t=float(life.data.time))
  held = {"on": True}
  monkeypatch.setattr(restaurant, "touched_by", lambda *a: held["on"])
  _sweep(life)
  assert _out(life), "under a hand"
  held["on"] = False
  _sweep(life)
  assert not _out(life)


def test_no_offer_lands_on_dishware_a_robot_touches(tmp_path, monkeypatch):
  """An offer sets the dishware out, so none lands while a robot touches
  it (`_worked_props`, the producer's `busy`); let go, it is offered and
  set out."""
  from pluggybot.economy.cadence import Cadence
  life = _life(tmp_path)
  # ...a rotation that offers it: the house's does not yet (cadence.json)
  life.producer = task_producer(life.tasks, HUMANOID_HOME, procedures=True,
                                cadence=Cadence._build(HUMANOID_HOME, {"kinds": {"set_table": {}}},
                                                       None))
  held = {"on": True}
  monkeypatch.setattr(restaurant, "touched_by", lambda *a: held["on"])

  def tables() -> list:
    for _ in range(8):
      life.data.time += 600.0
      life._task_step()
      offered = [t for t in life.tasks.tasks.values() if t.kind == "set_table"]
      if offered:
        return offered
    return []
  assert tables() == []
  held["on"] = False
  assert tables() and _out(life), "let go, offered and set out"


# ---- the grade ------------------------------------------------------------------------------


def test_the_place_laid_and_let_go_is_paid_for_its_time(tmp_path):
  """The grade on the seam: each piece on its spot at the call and still 5 s
  later, nothing touching it -- the table check's own verdict, banked on
  the job's own row, which pays for its time (Ben's call: the counter's 90
  for its 16 minutes, times the table's minutes over 16)."""
  life = _life(tmp_path)
  task = _offer(life)
  assert life._claim_task(task.id) and task.id in life._job_watches
  _lay(life)
  verdict = _grade(life, task.id)
  assert verdict["ok"], verdict["reason"]
  pay = default_table()["set_table"].base
  assert pay == round(90 * TABLE_MINUTES / HAND_MINUTES)
  assert life.ledger.robots[life.root]["balance"] == pay
  assert task.id not in life._job_watches


def test_a_free_piece_that_came_down_fast_fails_the_job(tmp_path):
  """A drop is a free piece's fall (`table.Watch`), from the claim: a cup
  knocked off and caught by nothing fails a place laid after it."""
  life = _life(tmp_path)
  task = _offer(life)
  assert life._claim_task(task.id)
  watch = life._job_watches[task.id]
  cup = life.model.jnt_dofadr[life.model.joint("cup_free").id]
  life.data.qvel[cup + 2] = -1.0
  watch()
  _lay(life)
  verdict = _grade(life, task.id)
  assert not verdict["ok"] and "the cup came down at 1.0 m/s" in verdict["reason"]


def test_a_fall_since_the_claim_fails_the_job(tmp_path):
  """...and ends it there (#701): the place stays out through the hold the
  fall's grade takes -- the job holds the table -- and is put away on the
  sweep after its verdict."""
  life = _life(tmp_path)
  task = _offer(life)
  assert life._claim_task(task.id)
  life.body.falls = 1
  _lay(life)
  life.body.run(life.body.hold_routine(1.0))
  _sweep(life)
  assert _out(life), "put away from under the job's grade"
  life.body.run(life.body.hold_routine(table.HOLD_S))
  verdict = life.tasks.get(task.id).verdict
  assert not verdict["ok"] and verdict["reason"].startswith("you fell"), verdict
  _sweep(life)
  assert not _out(life)


def test_a_table_job_held_past_its_minutes_is_failed_and_put_away(tmp_path):
  """#602's rule on the table: held past the time it gives with no `done`
  waiting on its grade, it is failed through its grade's door -- and the
  place is put away on the next sweep."""
  life = _life(tmp_path)
  task = _offer(life, minutes=0.01)
  assert life._claim_task(task.id)
  life.body.run(life.body.hold_routine(1.0))
  life._run_out_held()
  done = life.tasks.get(task.id)
  assert done.state == "failed" and "not graded within" in done.verdict["reason"]
  _sweep(life)
  assert not _out(life)


# ---- a restart ------------------------------------------------------------------------------


def test_a_restart_sets_an_open_offers_place_back_out_and_leaves_a_settled_one_away(tmp_path):
  """The world is compiled with the dishware put away: as the day begins
  (`begin` -> `restore_jobs`) the newest offer on the table, still open, has
  its place set out again with its draw's weights -- the continuation then
  puts the pieces where the world left them -- and one settled leaves it
  away, its weights still its draw's."""
  life = _life(tmp_path)
  task = _offer(life)
  k = task.secret["draw"]
  fresh = _life(tmp_path)                       # the same board, a world compiled anew
  assert not _out(fresh)
  fresh.begin((0.0, 0.0, 0.0))
  assert _out(fresh)
  plate = fresh.model.body("plate").id
  assert fresh.model.body_mass[plate] == pytest.approx(dw.draw(k).plate_kg)
  assert _at(fresh, "plate")[:2] == pytest.approx(ct.table_set_out(k).pieces["plate"][:2])
  fresh.tasks.resolve(task.id, table_verdict(), t=float(fresh.data.time))
  again = _life(tmp_path)
  again.begin((0.0, 0.0, 0.0))
  assert not _out(again)
  assert again.model.body_mass[again.model.body("plate").id] == pytest.approx(dw.draw(k).plate_kg)


def test_a_continuation_keeps_a_worked_place_where_the_world_left_it(tmp_path):
  """A restart is a continuation (#345): a cup moved since its offer comes
  back where the save had it, live; and an offer made between the save and
  a crash has its place set out for it (`set_out_unseen`)."""
  from pluggybot import continuation
  life = _life(tmp_path)
  task = _offer(life)
  cup = life.model.body("cup").id
  moved = _at(life, "cup") + np.array([0.05, -0.03, 0.0])
  dw._pose(life.model, life.data, cup, moved)
  mujoco.mj_forward(life.model, life.data)
  path = tmp_path / "world.npz"
  continuation.write(continuation.capture([life], life.world_fingerprint), path)
  back = _life(tmp_path, rebase=False)
  back.begin((0.0, 0.0, 0.0))
  got = continuation.restore([back], continuation.read(path))
  assert got["inPlace"] and "setOut" not in got
  assert _at(back, "cup") == pytest.approx(moved)
  assert back.model.geom_contype[back.model.body_geomadr[cup]] == 1, "live"
  # ...and saved before an offer, the world comes back without its set-out
  life.tasks.resolve(task.id, table_verdict(), t=float(life.data.time))
  _sweep(life)
  continuation.write(continuation.capture([life], life.world_fingerprint), path)
  late = _offer(life)                          # ...offered, and then the crash
  crashed = _life(tmp_path, rebase=False)
  crashed.begin((0.0, 0.0, 0.0))
  got = continuation.restore([crashed], continuation.read(path))
  assert got["setOut"] == [restaurant.TABLE] and _out(crashed)
  assert _at(crashed, "cup")[:2] == pytest.approx(
    ct.table_set_out(late.secret["draw"]).pieces["cup"][:2])


# ---- the world --------------------------------------------------------------------------------


def test_put_away_dishware_is_in_no_cast_and_meets_nothing():
  """Put away, a piece is in the parked group, which every cast leaves out
  (`perception.cull`) and no camera draws (MuJoCo's default groups), its
  contacts off and its weight held: a ray straight at it meets nothing,
  and the same ray cast at every group meets it."""
  model, data = _table_world()
  mujoco.mj_forward(model, data)
  plate = model.body("plate").id
  at = np.array(data.xpos[plate])
  hit = np.zeros(1, dtype=np.int32)
  below = at - np.array([0.0, 0.0, 0.3])
  up = np.array([0.0, 0.0, 1.0])
  assert mujoco.mj_ray(model, data, below, up, None, 1, -1, hit) > 0 and \
      model.geom_bodyid[hit[0]] == plate, "the probe ray meets it, cast at every group"
  for groups in (CAST_GROUPS, UNCULLED_GROUPS):
    mujoco.mj_ray(model, data, below, up, groups, 1, -1, hit)
    assert hit[0] < 0 or model.geom_bodyid[hit[0]] != plate, "it passes the plate"
  assert not mujoco.MjvOption().geomgroup[PARKED_GROUP], "no camera draws it"
  geoms = range(model.body_geomadr[plate], model.body_geomadr[plate] + model.body_geomnum[plate])
  assert not any(model.geom_contype[g] or model.geom_conaffinity[g] for g in geoms)
  assert model.body_gravcomp[plate] == 1.0 and at[2] < 0.0


def test_set_out_the_pieces_rest_on_the_table_and_put_away_they_wait_under_the_floor():
  """Set out, each piece rests where it was put, a second on: the body's
  broadphase masks are compiled from its geoms' put away, and switched with
  the geoms' alone the pieces fell through the table and the floor, 4 m in
  a second. Put away again, they wait under the floor, still."""
  model, data = _table_world()
  restaurant.set_out(model, data, restaurant.TABLE, 3)
  for _ in range(500):
    mujoco.mj_step(model, data)
  for p in dw.PIECES:
    assert data.xpos[model.body(p).id][2] == pytest.approx(ct.TABLE_H, abs=0.005), p
  restaurant.put_away(model, data, restaurant.TABLE)
  for _ in range(500):
    mujoco.mj_step(model, data)
  for p, (x, y) in restaurant.PARK_AT.items():
    assert data.xpos[model.body(p).id] == pytest.approx((x, y, dw.PARK_Z), abs=1e-6), p
  assert restaurant.away(model, data, restaurant.TABLE)


def test_a_redraw_is_the_place_compiled_over_that_draw():
  """`restaurant.redraw(k)` leaves the pieces' weights, inertias and friction
  where a world compiled over draw k has them, and the pinned statistic
  every camera's near plane is scaled by where it was."""
  base, data = _table_world(0)
  extent = float(base.stat.extent)
  restaurant.redraw(base, data, restaurant.TABLE, 7)
  want, _ = _table_world(7)
  for p in dw.PIECES:
    b, w = base.body(p).id, want.body(p).id
    assert base.body_mass[b] == pytest.approx(want.body_mass[w]), p
    assert base.body_inertia[b] == pytest.approx(want.body_inertia[w]), p
    for g, h in zip(range(base.body_geomadr[b], base.body_geomadr[b] + base.body_geomnum[b]),
                    range(want.body_geomadr[w], want.body_geomadr[w] + want.body_geomnum[w])):
      assert base.geom_friction[g][0] == pytest.approx(want.geom_friction[h][0]), p
  assert float(base.stat.extent) == extent


def test_the_served_table_is_the_set_outs():
  """One geometry, offline and live: the house's table 0 -- a top on legs,
  its south chair away -- is the one the table's set-out stands at, and
  the served dishware is the set-out's, piece for piece."""
  from pluggybot.humanoid import world as humanoid_world
  served = humanoid_world.home_spec().compile()
  offline = ct.model_of(ct.table_set_out(0))

  def table_geoms(m):
    b = m.body(restaurant.TABLE).id
    return sorted((m.geom(g).name, tuple(np.round(m.geom_size[g], 6)),
                   tuple(np.round(m.geom_pos[g], 6)))
                  for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]))
  assert table_geoms(served) == table_geoms(offline)
  assert len(table_geoms(served)) == 5, "a top and four legs"
  for m in (served, offline):
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{restaurant.TABLE}_chair_1") < 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{restaurant.TABLE}_chair_0") >= 0

  def piece(m, p):
    b = m.body(p).id
    return [(int(m.geom_type[g]), tuple(np.round(m.geom_size[g], 6)), tuple(np.round(m.geom_pos[g], 6)))
            for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])]
  for p in (*dw.PIECES, "place_card"):
    assert piece(served, p) == piece(offline, p), p
  assert restaurant.present(served) and not restaurant.present(offline)


# ---- the offer's words ------------------------------------------------------------------------


def test_the_sentence_says_the_graders_own_numbers_within_the_cap():
  """The template's numbers are the table check's constants, so neither
  moves without the other, and the sentence fits a description's cap -- the
  place laid from the pass's too, graded as the table's is."""
  from pluggybot.economy.tasks import MAX_DESCRIPTION
  for kind in ("set_table", "set_table_pass"):
    words = KINDS[kind].template
    for p, (a, b) in table.SPOTS.items():
      assert f"{100 * a:g} cm out" in words, (kind, p)
      if b:
        side = "left" if b > 0 else "right"
        assert re.search(rf"\b{100 * abs(b):g} cm (?:to your )?{side}\b", words), (kind, p)
    assert f"{100 * table.PLACE_M:g} cm" in words and f"{table.HOLD_S:g} s later" in words, kind
    assert len(words.format(minutes=999, offered=99_999_999)) <= MAX_DESCRIPTION, kind


#: The kinds whose place is laid at the table, each by its set-out.
TABLE_KINDS = list(restaurant.SET_OUTS)


def _offered() -> set:
  """Every kind a world's cadence offers, its own block's or the default's,
  off the committed `cadence.json` -- never a cached world's nor
  `$PLUGGY_CADENCE`'s."""
  import json
  from pluggybot.economy import cadence
  worlds = ["", *json.loads(cadence.CADENCE_PATH.read_text())["worlds"]]
  return {k for w in worlds for k in cadence.Cadence.load(w, cadence.CADENCE_PATH).kinds}


def test_the_tables_words_say_which_left_the_graders_is():
  """#723: the grader's left is the diner's (`table.place_frame`), the left
  of one who faces the card's face, as the robot laying the place does; so
  each row that states the place says each side as the robot's own, and so
  does each offered kind's sentence as it is offered -- the side read off the
  grader's frame and `SPOTS`, never typed (why: #723's period in
  Observatory.md)."""
  from pluggybot.mind.overseer import Menu
  assert {k for k, kind in KINDS.items() if kind.target_kind == "table"} == set(TABLE_KINDS), \
    "every kind at the table has a set-out"
  model, data = _table_world()
  so = ct.table_set_out(0)
  restaurant.set_out(model, data, restaurant.TABLE, so.k)
  mujoco.mj_forward(model, data)
  _, toward, left, _ = table.place_frame(model, data)
  # one who faces the card's face looks against `toward`: its left is theirs
  assert np.cross(-toward, left)[2] == pytest.approx(1.0), "a quarter turn anticlockwise"
  facing = np.array([math.cos(so.start[2]), math.sin(so.start[2]), 0.0])
  assert facing @ toward == pytest.approx(-1.0), "the robot at the place faces the card's face"
  rewards = default_table()
  served = {r["task"]: r["detail"] for r in rewards.as_context(
    challenges=True, kinds=Menu.for_world(HUMANOID_HOME).kinds)}
  assert served["set_table"] == rewards["set_table"].detail
  sides = {p: ("left" if b > 0 else "right", f"{100 * abs(b):g}")
           for p, (_, b) in table.SPOTS.items() if b}
  offered = _offered()
  for kind in TABLE_KINDS:
    row = rewards[kind].detail
    assert "as you face the card" in row, kind
    for p, (side, cm) in sides.items():
      assert f"{cm} cm to the line's {side} (your {side}" in row, (kind, p)
    if kind in offered:
      said = task_producer(TaskBoard(), HUMANOID_HOME, procedures=True)._offer(
        kind, restaurant.TABLE, 0.0).description
      for p, (side, cm) in sides.items():
        assert f"{cm} cm to your {side}" in said, (kind, p)


#: "Near the card", m: an offer here sets each piece out within this of the
#: card's foot (0.33 at the farthest a set-out draws, `counter.TABLE_SOURCES`).
NEAR_CARD_M = 0.35


def test_the_places_directions_say_where_an_offer_sets_the_pieces_out():
  """#723: the place's directions say where an offer there sets the pieces
  out -- on the table near the card, as the practice's job says they wait
  (`practice.TABLE_JOB`) -- and that none is where it goes: true of every
  kind a world offers at the table, over 32 set-outs, by the grader's own
  measure. The directions are the area's, so every kind's offer there
  carries them: one that sets its pieces out elsewhere (the pass's, #682's
  stage C, whose row says where) is offered by no world, and offering it
  moves these words (why: #723's period in Observatory.md)."""
  offered = _offered()
  here = [k for k in TABLE_KINDS if k in offered]
  assert "set_table" in here
  model, data = _table_world()
  for kind in here:
    words = task_producer(TaskBoard(), HUMANOID_HOME, procedures=True)._offer(
      kind, restaurant.TABLE, 0.0).as_context(0.0)["directions"]
    assert ("the plate, the cup and the fork are set out on the table near the card, "
            "none of them where it goes") in words, kind
    for k in range(32):
      restaurant.set_out(model, data, restaurant.TABLE, k, kind=kind)
      mujoco.mj_forward(model, data)
      got = table.measure(model, data)
      origin, toward, left, _ = table.place_frame(model, data)
      for p in dw.PIECES:
        rel = data.xpos[model.body(p).id] - origin
        assert abs(got[p]["aboveM"]) <= table.RESTING_M, (kind, k, p, "on the table")
        assert math.hypot(rel @ toward, rel @ left) <= NEAR_CARD_M, \
          (kind, k, p, "set out further from the card than near it: the words move with it")
        assert got[p]["offM"] > table.PLACE_M, (kind, k, p, "where it goes")


# ---- the place laid from the pass (#682's stage C, easy mode) ------------------------------


def _offer_pass(life, **params):
  """The place laid from the pass, offered as a cadence would offer it --
  none does (`cadence.json`): it is never offered."""
  maker = task_producer(life.tasks, HUMANOID_HOME, procedures=True)
  made, secret = maker._build("set_table_pass", restaurant.TABLE)
  return life.tasks.offer("set_table_pass", restaurant.TABLE, params={**made, **params},
                          secret=secret, ttl=3000.0, t=float(life.data.time))


def test_a_pass_offer_sets_its_pieces_out_on_the_pass_and_the_card_at_the_table(tmp_path):
  """The kind chooses the set-out (`restaurant.SET_OUTS`): an offer of the
  place laid from the pass sets each piece out where `pass_set_out` lays it,
  on the pass's top, its hidden weight the draw's, and the card at the
  table -- never the table's set-out beside the place."""
  life = _life(tmp_path)
  task = _offer_pass(life)
  k = task.secret["draw"]
  so, load = ct.pass_set_out(k), dw.draw(k)
  for p, (x, y, _) in so.pieces.items():
    assert _at(life, p) == pytest.approx((x, y, ct.PASS_H)), p
    assert life.model.body_mass[life.model.body(p).id] == pytest.approx(getattr(load, f"{p}_kg"))
  assert _at(life, "place_card") == pytest.approx((*ct.table_set_out(k).card[:2], ct.TABLE_H))
  assert life._target_activity(restaurant.TABLE).load == load


def test_a_restart_sets_an_open_pass_offer_back_out_on_the_pass(tmp_path):
  """`restore_jobs` sets the newest open offer's place out by ITS kind: a
  pass's pieces back on the pass, never beside the table."""
  life = _life(tmp_path)
  task = _offer_pass(life)
  k = task.secret["draw"]
  fresh = _life(tmp_path)
  fresh.begin((0.0, 0.0, 0.0))
  assert _out(fresh)
  for p, (x, y, _) in ct.pass_set_out(k).pieces.items():
    assert _at(fresh, p) == pytest.approx((x, y, ct.PASS_H)), p


def test_the_pass_set_out_lies_on_the_counter_in_a_row_each_draw_its_own():
  """Each piece rests whole on the expo counter's top, NEAR ITS DINING EDGE
  -- a carried piece hangs under the pass's top 0.32-0.35 m ahead of the
  soles, so the walk's hand-over clears the counter only 0.37 m off its face,
  and standing there a hand takes a piece no further than 0.45-0.47 ahead
  (set out 0.18-0.20 m in, the hand-over swept the cup into the counter's
  edge) -- the cup right of the plate and the fork left as a robot faces the
  pass; seeded, each draw its own, the fork's head in."""
  hx, hy, hz = hw.EXPO_HALF
  cx, cy = hw.BACK_X, sum(hw.EXPO_Y) / 2
  # each piece's reach behind its middle and ahead of it, the fork's turn spared
  r = {"plate": (dw.PLATE_R,) * 2, "cup": (dw.CUP_R,) * 2,
       "fork": (dw.HANDLE[0] + 0.005, dw.HANDLE[0] + 2 * dw.HEAD[0] + 0.005)}
  assert ct.PASS_H == pytest.approx(2 * hz)
  seen = set()
  for k in range(20):
    so = ct.pass_set_out(k)
    assert so.surface == ct.PASS_H
    for p, (x, y, yaw) in so.pieces.items():
      assert cx - hx + r[p][0] <= x and x + r[p][1] <= cx + hx, (k, p)
      assert abs(y - cy) + r[p][1] <= hy, (k, p)
      assert x < hw.BACK_X - hw.WALL_HALF_T, (k, p, "the dining side")
      assert x - (cx - hx) <= 0.12, (k, p, "near the dining edge")
    assert so.pieces["cup"][1] < so.pieces["plate"][1] < so.pieces["fork"][1], k
    assert abs(math.remainder(so.pieces["fork"][2], 2 * math.pi)) <= math.radians(6.0), k
    assert so.card == ct.table_set_out(k).card and so.load == dw.draw(k)
    seen.add(tuple(np.round(so.pieces["plate"], 6)))
  assert len(seen) == 20
  assert ct.pass_set_out(3) == ct.pass_set_out(3)


def test_the_place_laid_from_the_pass_is_graded_as_the_tables_and_pays_nothing(tmp_path):
  """Its grade is the table's evaluator under a row of its own (`HAND_JOBS`),
  paying nothing, shown to no mind -- and no world's cadence offers it."""
  import json
  from pathlib import Path
  from pluggybot.economy import scoring
  from pluggybot.economy.tasks import HAND_JOBS
  assert HAND_JOBS["set_table_pass"] == "table"
  assert scoring.EVALUATORS["set_table_pass"] is table.eval_table
  assert scoring.SAMPLERS["set_table_pass"] is table.sample_table
  row = default_table()["set_table_pass"]
  assert row.base == 0 and row.bonus == 0 and row.shown is False
  from pluggybot.economy.cadence import CADENCE_PATH
  cadence = json.loads(Path(CADENCE_PATH).read_text())
  for world, spec in {"default": cadence["default"], **cadence["worlds"]}.items():
    assert "set_table_pass" not in (spec.get("kinds") or {}), world
  life = _life(tmp_path)
  task = _offer_pass(life)
  assert life._claim_task(task.id) and task.id in life._job_watches
  _lay(life)
  verdict = _grade(life, task.id)
  assert verdict["ok"], verdict["reason"]
  assert life.ledger.robots[life.root]["balance"] == 0


def test_a_piece_on_the_pass_reads_lifted_off_the_pass_never_the_table(tmp_path):
  """`<piece>Lifted` is a piece over where it was SET OUT (#682's stage C,
  its review): read off the table's top, a piece on the pass, 0.18 m over
  it, read lifted the moment it was set out, every flag on the wire false.
  Where each rests is read once it is out, and kept across a restart."""
  import json
  life = _life(tmp_path)
  task = _offer_pass(life)
  act = life._target_activity(restaurant.TABLE)
  mujoco.mj_forward(life.model, life.data)
  act.sense(life.model, life.data)
  assert act.flags["setOut"] is True
  assert not any(act.flags[f"{p}Lifted"] for p in dw.PIECES), act.flags
  cup = life.model.jnt_qposadr[life.model.joint("cup_free").id]
  life.data.qpos[cup + 2] += 2 * dw.LIFT_ON
  mujoco.mj_forward(life.model, life.data)
  act.sense(life.model, life.data)
  assert act.flags["cupLifted"] is True and act.flags["plateLifted"] is False
  again = restaurant.activity_for(life.model, life.data, restaurant.TABLE, task.secret["draw"])
  again.restore_kept(json.loads(json.dumps(act.kept_state())))
  assert again.rest == pytest.approx(act.rest) and again.lifted["cup"].value is True
  # ...and the table's set-out reads off the table's top, as it always did
  life.tasks.resolve(task.id, table_verdict(), t=float(life.data.time))
  _offer(life)
  mujoco.mj_forward(life.model, life.data)
  act.sense(life.model, life.data)
  assert act.rest == pytest.approx({p: ct.TABLE_H for p in dw.PIECES}, abs=1e-6)


def test_a_place_set_out_in_the_hook_reads_where_it_was_set_out(tmp_path):
  """The served order (#682's stage C, its second review): an offer sets the
  place out in the physics hook (`_task_step`, its producer's tick) and the
  activities' hook senses it in the SAME pass, before a forward pass has
  carried the new poses into `xpos` -- read off `xpos` there, every piece
  rested 1 m under the floor, where it is put away, and every `<piece>Lifted`
  of every table job read true the whole job. Each kind's own surface."""
  life = _life(tmp_path)
  act = life._target_activity(restaurant.TABLE)
  for offer, surface in ((_offer, ct.TABLE_H), (_offer_pass, ct.PASS_H)):
    mujoco.mj_step(life.model, life.data)        # a step with the place put away
    task = offer(life)                           # ...the hook sets it out
    act.sense(life.model, life.data)             # ...and senses it, the same pass
    for _ in range(3):
      mujoco.mj_step(life.model, life.data)
      act.sense(life.model, life.data)
    assert act.rest == pytest.approx({p: surface for p in dw.PIECES}, abs=1e-6), task.kind
    assert not any(act.flags[f"{p}Lifted"] for p in dw.PIECES), (task.kind, act.flags)
    life.tasks.resolve(task.id, table_verdict(), t=float(life.data.time))
    restaurant.put_away(life.model, life.data, restaurant.TABLE)
    act.sense(life.model, life.data)


# ---- put away, the pieces stay put away ------------------------------------


def test_put_away_after_any_set_out_the_pieces_stay_where_they_are_parked(tmp_path):
  """Put away, a piece's weight is held (`gravcomp`), and MuJoCo holds it
  only while the model COUNTS a body so held -- a count `mj_setConst`, the
  end of every set-out's redraw, takes afresh. Taken with the place set out
  it was 0, and from then on the parked pieces fell at g and the lifecycle
  put them back, and said so, every second (Observatory.md, the period of
  `loop-k4`): the pair hooks one board twice, so its place is set out twice,
  and a counter prop's set-out while the table's is out does it for a robot
  alone."""
  from pluggybot.home import storeroom

  def stays(model, data) -> bool:
    restaurant.put_away(model, data, restaurant.TABLE)
    for _ in range(int(round(0.2 / model.opt.timestep))):
      mujoco.mj_step(model, data)
    return restaurant.away(model, data, restaurant.TABLE)

  # one offer's place set out twice, as the pair's two hooks do
  model, data = _table_world()
  restaurant.set_out(model, data, restaurant.TABLE, 3)
  restaurant.set_out(model, data, restaurant.TABLE, 3)
  assert stays(model, data), dw._placed_at(model, data, model.body("plate").id)
  assert model.ngravcomp == len(dw.PIECES)
  # ...a counter prop set out while the place is out (the house with both)
  spec = mujoco.MjSpec.from_string(STUB_WORLD)
  spec.texturedir = "models/"
  storeroom.attach(spec)
  restaurant.attach(spec)
  model = spec.compile()
  data = mujoco.MjData(model)
  restaurant.set_out(model, data, restaurant.TABLE, 3)
  storeroom.set_out(model, data, "drawer", 3)
  assert stays(model, data) and model.ngravcomp == len(dw.PIECES)
  # ...and through the lifecycle, every way a place is set out again -- the
  # pair's second hook, a restart's, a continuation's -- the put-away is
  # said once, not at every sweep
  life = _life(tmp_path)
  said = []
  life.say_hooks.append(lambda t, msg, *a: said.append(msg))
  task = _offer(life)
  life._set_out_job(task)
  life.restore_jobs()
  assert life.set_out_unseen(task.number - 1) == [restaurant.TABLE]
  life.tasks.expire_due(1e9)
  for _ in range(4):
    _sweep(life)
    for _ in range(int(round(1.0 / life.model.opt.timestep))):
      mujoco.mj_step(life.model, life.data)
  assert sum("things are put away" in m for m in said) == 1, said
  assert not _out(life)


def test_the_models_constants_are_recomputed_only_through_set_const():
  """The fence round it, as `CTRL_WRITERS` is `data.ctrl`'s: a bare
  `mj_setConst` anywhere in `src/` takes the count with whatever is parked
  just then -- both redraws called it so, and the put-away held nothing."""
  import pathlib

  from pluggybot.home import storeroom
  src = pathlib.Path(dw.__file__).parents[1]
  callers = sorted(str(p.relative_to(src)) for p in src.rglob("*.py")
                   if "mj_setConst(" in p.read_text())
  assert callers == ["activity/dishware.py"], callers
  for module in (restaurant, storeroom):
    assert "set_const(model)" in pathlib.Path(module.__file__).read_text(), module.__name__


# ---- the board's own lines, and a robot's status ---------------------------


def test_the_other_robot_is_shown_a_robots_own_last_line_never_the_boards(tmp_path,
                                                                         monkeypatch):
  """`others[].doing` was the other robot's `status`, its lifecycle's last
  narrated line -- and both lifecycles narrate the BOARD's doings (an offer,
  its things set out, an expiry, things put away), so a robot standing idle
  was shown as doing a job it had never taken (Observatory.md, the period
  of `loop-k4`). `status` is as it was, on the wire and under a portrait;
  the other robot is shown the last line that was the robot's OWN
  (`HubLifecycle.doing`; since #709 only a line its call marks so),
  whichever of the board's eight lines came since."""
  from types import SimpleNamespace

  from pluggybot.lifecycle import others_context
  life = _life(tmp_path)
  life.producer = task_producer(life.tasks, HUMANOID_HOME, procedures=True)
  said = []
  life.say_hooks.append(lambda t, msg, *a: said.append(msg))

  def shown() -> str:
    return others_context(SimpleNamespace(peers=[life]))[0]["doing"]

  def the_board_says(*phrases, by) -> None:
    """`by()` makes the board say each of `phrases`: narrated, the status
    its newest line as ever, and `doing` what it was."""
    before, n = shown(), len(said)
    by()
    new = said[n:]
    for phrase in phrases:
      assert any(phrase in m for m in new), (phrase, new)
    assert all(any(p in m for p in BOARD_LINES) for m in new), new
    assert life.status == new[-1] and shown() == before

  # a robot that has said nothing of its own yet: an offer, for a prop this
  # stub world has not
  assert (life.status, shown()) == ("", "")
  life.data.time = 5000.0
  the_board_says(" offered: ", "to set out", by=lambda: _sweep(life))
  assert shown() == ""
  life._say("EXPLORE: heading for the kitchen", doing=True)
  assert life.status == shown() == "EXPLORE: heading for the kitchen"
  # the table's offer sets its things out; a continuation leaves them under
  # a hand; the offer is claimed by a robot that is not here, and a restart
  # failed a job; then every offer lapses and the place is put away
  box = {}
  the_board_says("is set out", by=lambda: box.update(task=_offer(life)))
  task = box["task"]
  monkeypatch.setattr(life, "_worked_props", lambda: {restaurant.TABLE})
  the_board_says("left as the world was saved",
                 by=lambda: life.set_out_unseen(task.number - 1))
  monkeypatch.undo()
  life.tasks.claim(task.id, robot="r9_gone", t=float(life.data.time))
  the_board_says("whoever held it is not", by=life._held_jobs)
  life.tasks.interrupted = [life.tasks.get(task.id)]

  def a_restart_failed_one() -> None:
    for gone in life.tasks.announce_interrupted(t=float(life.data.time)):
      life._say(f"TASK {gone.id} ({gone.kind}): interrupted by a restart")
  the_board_says("interrupted by a restart", by=a_restart_failed_one)

  def lapse() -> None:
    life.data.time = 1e9
    _sweep(life)
  the_board_says(" expired: ", "things are put away", by=lapse)
  assert shown() == life.doing == "EXPLORE: heading for the kitchen"
  # ...and none of the eight is marked the robot's doing where the lifecycle
  # says it (#709: a line is private unless its call marks it)
  import ast
  import inspect

  from pluggybot import lifecycle as lc
  marked = {}
  for node in ast.walk(ast.parse(inspect.getsource(lc))):
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_say" and node.args):
      text = ast.unparse(node.args[0])
      for phrase in BOARD_LINES:
        if phrase in text:
          marked.setdefault(phrase, []).append(
            any(k.arg == "doing" for k in node.keywords))
  assert set(marked) == set(BOARD_LINES), sorted(set(BOARD_LINES) - set(marked))
  assert not any(any(v) for v in marked.values()), {k: v for k, v in marked.items() if any(v)}


#: The board's own lines, by a phrase of each as the lifecycle says it: an
#: offer, an expiry, an offer's things set out (or not there to set out, or
#: left where a hand is on them), things put away, a claim given back for a
#: robot that is not in this world, a job a restart failed.
BOARD_LINES = (" offered: ", " expired: ", "is set out", "to set out",
               "left as the world was saved", "things are put away",
               "whoever held it is not", "interrupted by a restart")
