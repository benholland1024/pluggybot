"""#560 stage (c2)'s instrument (`scripts/putdown_spike.py`): each lever is
measured as #620's instrument measures a carry, so its base lever must BE
our procedure's carry as #620 measured it, and its copies of #620's
measures must be #620's -- or the levers are compared with something other
than the window SimNotes reports ("The table's put-downs, measured")."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import putdown_spike as ps  # noqa: E402
import waist_spike as ws  # noqa: E402

from pluggybot.humanoid import arms as ha  # noqa: E402
from pluggybot.humanoid import reach as rc  # noqa: E402


def test_the_base_lever_is_our_procedures_carry_where_the_window_stands_it():
  for piece, (side, pick, put, _) in ws.PIECES.items():
    c = ps.carries(ps.LEVERS["base"], piece)[0]
    assert c.side == side
    assert np.allclose(c.pick, pick, atol=1e-12) and np.allclose(c.put, put, atol=1e-12), piece
    assert np.allclose(c.rot, ws.GRIPS["t45"]) and np.allclose(c.put_rot, ws.GRIPS["t45"]), piece
    for dx, dy in ((0.0, -0.26), (0.05, 0.13)):
      st = ps.Stance(dx, dy, 0.0)
      # ...placed as `ws.window_tasks` places a stance's points
      assert list(st.point(c.pick)) == [round(pick[0] - dx, 4), round(pick[1] - dy, 4), pick[2]]
      n, c_ = st.slab()
      assert np.allclose(n, (1.0, 0.0)) and c_ == round(ws.TABLE_GAP - dx, 4)


def test_a_turned_stance_sees_the_table_where_the_start_put_it():
  st = ps.Stance(0.04, -0.11, 25.0)
  n, c = st.slab()
  for y in (-0.3, 0.0, 0.4):
    # a point on the table's edge, and one 5 cm in
    for x, inside in ((ws.TABLE_GAP, 0.0), (ws.TABLE_GAP + 0.05, 0.05)):
      p = st.point((x, y, 0.72))
      assert abs(float(n @ p[:2]) - c - inside) < 2e-4, (x, y)
  # ...and turns a grip with it: a turned stance's fingers keep their way in the world
  r = st.rot(ws.GRIPS["t45"])
  back = np.array([[math.cos(math.radians(25.0)), -math.sin(math.radians(25.0)), 0.0],
                   [math.sin(math.radians(25.0)), math.cos(math.radians(25.0)), 0.0],
                   [0.0, 0.0, 1.0]]) @ r
  assert np.allclose(back, ws.GRIPS["t45"])


def _posed_into_the_table():
  sm, table, _ = ws._models("none")
  # the deepest bow but one, its trunk into the top from the nearer edge
  # 8 cm and from the further 3 (`into_table`)
  p = next(q for q in table if q.label() == "(-0.25, 0.95)")
  ws.pose(sm, p)
  q, _, _ = rc.solve_seeded(sm, "left", np.array([0.30, 0.30, 0.70]), ws.GRIPS["down"])
  sm._set_arm("left", q)
  return sm


def test_the_slab_is_620s_table_where_the_stance_stands_square():
  sm = _posed_into_the_table()
  seen = set()
  for edge in (0.13, 0.18):
    into = ws.into_table(sm, edge)
    seen.add(round(into, 3))
    assert into > 0.0
    assert ps.into_slab(sm, (np.array([1.0, 0.0]), edge)) == into
  assert len(seen) == 2


def test_the_first_seed_that_reaches_is_the_one_the_seeded_solve_keeps():
  sm, table, _ = ws._models("none")
  ws.pose(sm, table[0])
  for pos, grip in (((0.42, -0.31, 0.78), "t45"), ((0.30, 0.30, 0.80), "down"),
                    ((0.90, 0.20, 0.80), "t45")):
    for side in ("left", "right"):
      a = ps.solve_first(sm, side, np.array(pos), ws.GRIPS[grip])
      b = rc.solve_seeded(sm, side, np.array(pos), ws.GRIPS[grip])
      assert np.array_equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2], (pos, side)


def test_one_posture_is_620s_carry_and_a_turn_comes_first_on_a_line():
  # the plate from our procedure's step, where #620's carry finds the deep bow
  # that dips into the table, and from 0.30, where it carries clear; the cup
  # 5 cm to the right, carried only at the served carry's looser tolerance
  sm, table, reach_m = ws._models("none")
  for piece, dy in (("plate", -0.26), ("plate", -0.30), ("cup", -0.05)):
    side, pick, put, _ = ws.PIECES[piece]
    a, b = np.array(pick) - (0.0, dy, 0.0), np.array(put) - (0.0, dy, 0.0)
    got = ws.plan(sm, side, ha.HandGoal(a, ws.GRIPS["t45"]), table, reach_m)
    rot = got[4]
    order = [q for q in table if q.label() == got[0].label()] + \
      [q for q in table if q.label() != got[0].label()]
    old = ws.carry_plan(sm, side, order, ws.carry_points(a, b), rot, reach_m, edge=ws.TABLE_GAP)
    c = ps.Carry(piece, side, a, b, rot, rot)
    top, moves, _ = ps.carry_points(c, ps.LEVERS["base"], rot)
    new = ps.follow_one(sm, side, order, [(top, rot)] + moves[1] + moves[2], reach_m,
                        ps.Stance(0.0, dy, 0.0).slab())
    assert (old is None) == (new is None) == (piece == "cup"), (piece, dy)
    if old is not None:
      assert old[0].label() == new[0].label() and np.array_equal(old[1], new[1])
      assert (old[3], old[4]) == (new[2], new[3])
  # a carried command turns at `HAND_TURN` while it walks at `HOLD_SPEED`
  r1 = ws.GRIPS["down"]
  pts = ps.line((0.3, 0.2, 0.8), (0.5, 0.2, 0.8), ws.GRIPS["t45"], r1)
  turned = math.pi / 4 * rc.HOLD_SPEED / rc.HAND_TURN
  for x, r in pts:
    done = np.allclose(r, r1)
    assert done == (x[0] - 0.3 >= turned - 1e-9), x


def test_the_flown_table_scenes_are_the_jobs_procedure_with_the_levers_lines_alone():
  """#560 stage (c2) flew our table procedure changed by its levers alone
  (`hands_check.SCENES`, SimNotes "The table's put-downs, measured"): each
  scene compiles, grades as the table, and differs from the job's procedure
  in the plate's grip point, its put-down, the grip's turn and the step --
  a line of the job moved under it is a scene that no longer flies the
  lever, and fails here."""
  from pluggybot.evaluation import hands_check as hc
  job = hc.TABLE.splitlines()
  for name in ("table_rim30", "table_rim30_t60"):
    grader, source = hc.SCENES[name]
    assert grader == "table"
    hc.compiled(name)
    mine = source.splitlines()
    assert len(mine) == len(job)
    changed = [a.split("(")[0].split("=")[0].strip() for a, b in zip(job, mine) if a != b]
    assert changed == ["px", "py", "step", "reach", "move_hand"], (name, changed)


def test_a_pull_takes_another_posture_onward_never_one_it_holds_or_has_left():
  """The served measure's pulls (the lift, the set-down) take another
  posture where the arm misses, as `Reach._pulled` does
  (`posture_for(onward=True)`): from the arm as it is alone, and never the
  posture held nor one left this move -- where a straight carry's may go
  back to any (`posture_for`)."""
  sm, _, reach_m = ws._models("none")
  hold = ps._hold("none")
  side, _, put, _ = ws.PIECES["plate"]
  x = np.array(put) + (0.0, 0.28, ws.LIFT_M)
  rot = ws.GRIPS["t45"] @ np.diag([-1.0, -1.0, 1.0])
  q = rc.seed(side, rc.SEEDS[1])
  first = ps.posture_for(sm, side, hold, "", q, x, rot, reach_m)
  assert first is not None
  gone = {first.label()}
  nxt = ps.posture_onward(sm, side, hold, gone, q, x, rot, reach_m)
  assert nxt is None or nxt.label() not in gone
  assert ps.posture_onward(sm, side, hold, {p.label() for p in hold}, q, x, rot, reach_m) is None
