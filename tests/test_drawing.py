"""Drawing on a whiteboard with the quadruped's arm (issue #406; SimNotes,
"Drawing on legs"). Every rule pinned without flying a drawing: the pen's
module against the rack's and the arm's envelopes, the arm's reach from
the floor, and the plotter's logic against a PERFECT ARM -- a world of the
pen module and a board, the module put exactly where the arm is aimed --
in a second, where a flown figure is minutes. The flown figures are
`scripts/draw_spike.py`'s tables.
"""

import ast
import dataclasses
import inspect
import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from pluggybot import tick
from pluggybot.legs import arm as am
from pluggybot.legs import rack as rk
from pluggybot.legs.draw import BoardWork
from pluggybot.legs.model import CHOSEN
from pluggybot.legs.swap import ToolSwap
from pluggybot.rack.coupling import PEG_ABOVE_BODY
from pluggybot.tools import drawing as dw
from pluggybot.tools import strokes


# ---- the pen's module ----------------------------------------------------------------


def _pen_alone(mount_x: float = rk.PEN_MOUNT_X):
  """The pen's module on its own, hung by its peg's axis at the origin."""
  face = rk.pen_face().replace(f'pos="{rk._v(rk.PEN_MOUNT_X, 0, rk.PEN_RAIL_Z)}"',
                               f'pos="{rk._v(mount_x, 0, rk.PEN_RAIL_Z)}"')
  xml = (f'<mujoco><compiler angle="radian"/><default>{rk.tool_default("module_pen")}'
         f'</default><worldbody>'
         + rk.tool_xml("module_pen", (0, 0, 1.0), mass=rk.TOOL_KG["module_pen"] - rk.PEN_PARTS_KG,
                       face=face)
         + f'</worldbody><actuator>{rk.tool_actuators_xml()}</actuator></mujoco>')
  m = mujoco.MjModel.from_xml_string(xml)
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  return m, d


def _lean_deg(m, d) -> float:
  """How far off plumb the module hangs from its peg: its CoM's angle
  under the peg's axis."""
  b = m.body("module_pen").id
  com = d.subtree_com[b] - (d.xpos[b] + [0, 0, PEG_ABOVE_BODY])
  return math.degrees(math.atan2(abs(com[0]), -com[2]))


def test_the_pen_hangs_plumb_on_its_peg_and_fits_the_arms_envelope():
  """The module's CoM is under its peg (the rack's `on_bay` asks 2 deg):
  the rover's carriage, 26 mm ahead of its plate, hung it 16 deg off. And
  it is a tool the arm may carry (`legs.arm`'s envelope)."""
  m, d = _pen_alone()
  assert _lean_deg(m, d) < 1.0
  assert _lean_deg(*_pen_alone(mount_x=-(rk.TOOL_HALF_X + 0.016))) > rk.HUNG_TILT_DEG, \
    "the premise: the rover's layout tips it"
  b = m.body("module_pen").id
  assert m.body_subtreemass[b] == pytest.approx(rk.TOOL_KG["module_pen"])
  assert m.body_subtreemass[b] <= am.TOOL_MAX_KG
  tip = d.site_xpos[m.site(rk.PEN_TIP).id] - (d.xpos[b] + [0, 0, PEG_ABOVE_BODY])
  assert -tip[2] <= am.TOOL_MAX_DROP_M
  # the plotter's drawing of its point off the fork is the module's own:
  # the tip site, and the shaft's rounded end past it
  ahead, up = dw.tip_from_vertex()
  seat = am.ArmSpec().fork.seat_rise()
  assert m.geom(rk.PEN_SHAFT).size[0] == rk.PEN_TIP_R
  assert (ahead, up) == pytest.approx((-tip[0] + rk.PEN_TIP_R, tip[2] + seat))


def test_the_carriage_is_the_slide_the_bill_buys():
  """The pen's carriage is an Actuonix L12-100 (Parts.md): its 100 mm
  stroke, its 22 N, its 25 mm/s -- the bill's part, not the rover's 110 mm."""
  m, _ = _pen_alone()
  a = m.actuator(rk.PEN_ACTUATOR)
  assert tuple(a.ctrlrange) == pytest.approx((-0.050, 0.050))
  assert tuple(a.forcerange) == pytest.approx((-22.0, 22.0))
  assert rk.PEN_SPEED <= 0.025 and dw.DRAW_SPEED <= rk.PEN_SPEED


def test_every_figure_fits_and_an_answer_is_never_shrunk():
  """A figure on the pen's menu is fitted into the board's envelope; an
  answer fits as it is written, because the evaluator compares the ink
  against the glyphs at their own size (`questions.ANSWER_CAP`)."""
  from pluggybot.lifecycle import QUAD_HOME, figure_program
  env = dw.Envelope.for_board(dw.Board.from_meta(_boards()["whiteboard_a"]))
  assert env.size == pytest.approx((0.10, 0.20))
  for name in ("house", "tree", "sun", "robot", "square", "circle"):
    assert figure_program(QUAD_HOME, "whiteboard_a", name).fits(env), name
  for n in range(100):
    prog = figure_program(QUAD_HOME, "whiteboard_a", f"answer:{n}")
    assert prog.fits(env) and prog.strokes == strokes.program("answer", text=str(n)).strokes


def _boards() -> dict:
  from pluggybot.home import world as home
  return home.BOARDS


def test_the_arm_reaches_every_corner_of_the_board_from_the_floor():
  """Lying (the torso's centre `belly_depth` up, `LIE_M` out of the wall,
  the board's face `BOARD_PROUD_M` nearer) the fork reaches the envelope's
  four corners, pressed and lifted, with the pen's carriage at either end:
  from the floor, not only the middle."""
  from pluggybot.legs.draw import LIE_M
  face = LIE_M - dw.BOARD_PROUD_M
  ahead, up = dw.tip_from_vertex()
  spec = am.ArmSpec()
  board = dw.Board.from_meta(_boards()["whiteboard_a"])
  env = dw.Envelope.for_board(board)
  z_mid = board.z - CHOSEN.belly_depth
  for h in (env.z_min, env.z_max):
    for past in (-dw.LIFT - dw.PROBE_CLEAR, dw.PRESS_EXTRA + dw.PROBE_PAST):
      x = face - ahead + past
      assert am.solve_vertex(spec, x, z_mid + h - up, near=am.CARRY_Q) is not None, (h, past)


# ---- the plotter, against a perfect arm ---------------------------------------------------


class _PerfectArm:
  """The arm as the plotter drives it, tracking its aim exactly: `goal` is
  (shoulder, forearm's absolute angle), `q()` reads it back."""

  def __init__(self, spec):
    qs, qe = am.CARRY_Q                   # a carried tool rides at the carry pose
    self.goal = np.array([qs, qs + qe])

  def aim(self, shoulder: float, elbow: float) -> None:
    self.goal = np.array([shoulder, shoulder + elbow])

  def q(self):
    return float(self.goal[0]), float(self.goal[1])


class _Handle:
  prefix = ""


class _Bench:
  """A torso at the origin, level, facing +x: the pen's module put where
  the fork's vertex is aimed (a mocap body), and a board `dist` ahead,
  turned `yaw_deg` about z -- a board the plotter knows nothing of until
  it touches it."""

  def __init__(self, dist: float = 0.60, yaw_deg: float = 2.0, board_z: float = 0.10):
    self.posture, self.falls, self.carrying = "lying", 0, "module_pen"
    self.face_x = dist
    self.arm_spec = am.ArmSpec()
    self.arm = _PerfectArm(self.arm_spec)
    self.handle = _Handle()
    yaw = math.radians(yaw_deg)
    # the board: its face `dist` ahead at its middle, its normal toward -x
    hx = 0.01
    self.board = dw.Board("board", dist + hx * math.cos(yaw), hx * math.sin(yaw), board_z,
                          (hx, 0.16, 0.13), yaw)
    q = f"{math.cos(yaw / 2)} 0 0 {math.sin(yaw / 2)}"
    xml = (f'<mujoco><compiler angle="radian"/><option timestep="0.002"/>'
           f'<default>{rk.tool_default("module_pen")}</default><worldbody>'
           f'<body name="board_body" pos="{self.board.x} {self.board.y} {board_z}" quat="{q}">'
           f'<geom name="board" type="box" size="{hx} 0.16 0.13" friction="0.25" priority="1"/>'
           f'</body><body name="module_pen" mocap="true" childclass="module_pen_tool">'
           f'<geom name="module_pen_body" type="box" size="{rk.TOOL_HALF_X} {rk.TOOL_HALF_Y} '
           f'{rk.TOOL_HALF_Z}" contype="0" conaffinity="0"/>{rk.pen_face()}</body>'
           f'</worldbody><actuator>{rk.tool_actuators_xml()}</actuator></mujoco>')
    self.model = mujoco.MjModel.from_xml_string(xml)
    self.data = mujoco.MjData(self.model)
    self._place()
    mujoco.mj_forward(self.model, self.data)
    self.commands: list = []

  _vertex_goal = ToolSwap._vertex_goal

  def _place(self) -> None:
    """The module seated on the fork's vertex, facing the robot."""
    x, z = self._vertex_goal()
    peg_z = z + self.arm_spec.fork.seat_rise()
    self.data.mocap_pos[0] = [x, 0.0, peg_z - PEG_ABOVE_BODY]
    self.data.mocap_quat[0] = [0.0, 0.0, 0.0, 1.0]           # yaw pi

  def _twist_routine(self, vx, vy, w):
    yield (vx, vy, w)

  # the stepper `tick.run` drives
  def step(self, command) -> None:
    self._place()
    mujoco.mj_step(self.model, self.data)
    self.commands.append((round(float(self.data.ctrl[0]), 9),
                          tuple(round(float(v), 9) for v in self.arm.goal)))

  def run(self, routine):
    return tick.run(self, routine)


def _estimate(bench, x_off: float = 0.0) -> dw.BoardEstimate:
  """Where the tags would put the board: its face's middle, off by `x_off`."""
  return dw.BoardEstimate(x_face=bench.board.x - bench.board.half[0] + x_off,
                          y_mid=0.0, z_mid=bench.board.z)


def test_the_board_is_found_by_touch_from_where_the_tags_put_it():
  """Four probes find the board's plane: its yaw to a tenth of a degree
  and its face to a millimetre, from an estimate 15 mm off."""
  bench = _Bench(yaw_deg=3.0)
  p = dw.PenPlotter(bench, bench.board)
  cal = bench.run(p.calibrate_routine(_estimate(bench, x_off=0.015)))
  assert cal["ok"] and cal["probes"] == 4
  assert cal["yawDeg"] == pytest.approx(3.0, abs=0.15)
  # ...the face at the carriage's middle: where the fork's vertex was when
  # the quill read the touch, plus the pen's point ahead of the vertex, less
  # the quill's travel by then
  face = p._x_at(0.0, cal["zHome"], 0.0) + p.tip_ahead - dw.QUILL_TOUCH
  assert face == pytest.approx(bench.face_x, abs=0.001)


def test_a_figure_is_inked_and_the_pen_lifts_between_strokes():
  """A two-stroke figure inks both strokes and lays NO ink travelling:
  the lift off and the press on are no travel rows (`draw_program_
  routine`), and at first they were, half of every figure -- the
  evaluator fails a figure past a quarter (`scoring.DRAW_MAX_TRAVEL_INK`)."""
  bench = _Bench()
  p = dw.PenPlotter(bench, bench.board)
  assert bench.run(p.calibrate_routine(_estimate(bench)))["ok"]
  prog = strokes.StrokeProgram("two", (((-0.02, 0.0), (0.02, 0.0)), ((0.0, -0.02), (0.0, 0.02))))
  used = bench.run(p.draw_program_routine(prog))
  assert used["drew"] and used["strokes_drawn"] == 2
  assert used["inked_fraction"] > 0.9 and used["travel_ink_fraction"] == 0.0
  assert used["form_rms_mm"] < 0.3


def test_the_plotter_steers_by_nothing_the_sim_alone_knows(monkeypatch):
  """CALIBRATION READS NO GROUND TRUTH: with the world's records of the
  tip -- where it is on the board, whether it touches -- answering
  nonsense, every command the plotter gives (the carriage, the arm's aim)
  is the same, step for step -- and with the WORLD's board (the record's,
  handed to the plotter) put somewhere else. The rover's calibration read
  the tip."""
  def fly(moved: bool = False):
    bench = _Bench(yaw_deg=1.0)
    board = (dataclasses.replace(bench.board, x=bench.board.x + 0.05, z=0.3, heading=1.0)
             if moved else bench.board)
    p = dw.PenPlotter(bench, board)
    bench.run(p.calibrate_routine(_estimate(bench, x_off=0.01)))
    bench.run(p.draw_program_routine(strokes.program("square", size=0.03)))
    return bench.commands

  truth = fly()
  monkeypatch.setattr(dw.PenPlotter, "pen_board", lambda self: (0.5, -0.5))
  monkeypatch.setattr(dw, "pen_on_board", lambda *a: True)
  assert fly() == truth
  assert fly(moved=True) == truth


def test_a_fall_ends_the_drawing_and_the_arm_is_aimed_no_more():
  """A body that fell -- its fall counter moved, the pen thrown -- folded
  the arm: the plotter stops the figure there (`stopped: fell`) and never
  aims the arm again -- aimed on, it held the fork out through the get-up
  as the swap's did before #405's review. ⚠ Whatever its posture says: a
  body up and lying again looks as it did."""
  bench = _Bench()
  p = dw.PenPlotter(bench, bench.board)
  assert bench.run(p.calibrate_routine(_estimate(bench)))["ok"]
  prog = strokes.program("square", size=0.03)

  aimed = None
  routine = p.draw_program_routine(prog)
  try:
    k = 0
    while True:
      cmd = next(routine)
      if k == 600:
        bench.falls, bench.carrying = 1, None      # `QuadMission._fall_check`
        aimed = bench.arm.goal.copy()
      bench.step(cmd)
      k += 1
  except StopIteration as done:
    used = done.value
  assert used.get("stopped") == "fell", used
  assert np.array_equal(bench.arm.goal, aimed), "aimed after the fall"


def test_the_steering_names_no_truth():
  """...and the fence behind it: what the plotter steers by reads none of
  the sim's own records of the tip or its contacts."""
  steering = ("quill", "vertex", "carriage", "_aim", "_goal_vertex", "move_routine",
              "_probe_routine", "calibrate_routine", "_x_at", "_targets", "_home_across",
              "draw_program_routine")
  banned = {"site_xpos", "xpos", "xmat", "contact", "ncon", "pen_board", "pen_on_board",
            "local", "_home", "_tip", "board", "trace", "inked_polyline", "error_stats"}
  for name in steering:
    tree = ast.parse(inspect.cleandoc("\n" + inspect.getsource(getattr(dw.PenPlotter, name))))
    seen = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    seen |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not seen & banned, (name, seen & banned)


# ---- the walk to the board, its routines stubbed -----------------------------------------


class _Walk(BoardWork):
  """`BoardWork.draw_routine`'s own `self` with the body's routines stubbed,
  each recorded by name in `did`: the walks arrive, `fall_on` names the one
  a fall throws the pen in (`QuadMission._fall_check`), `across` is how far
  off the board's axis each walk in stops, and `faces` what each look
  answers."""

  def __init__(self, fall_on: str = "", across: float = 0.0, faces=(True,)):
    self.data = SimpleNamespace(time=0.0)
    self.carrying, self.falls, self.posture = "module_pen", 0, "standing"
    self.pose, self.last_drive, self.working = (2.0, 0.0, math.pi), None, False
    self.fall_on, self.across, self.faces = fall_on, across, list(faces)
    self.did: list[str] = []
    self._init_draw()

  def _did(self, what: str) -> None:
    self.did.append(what)
    self.data.time += 1.0
    if what == self.fall_on:
      self.falls, self.carrying = self.falls + 1, None

  def board_pose(self, name, dist):
    return (dist, 0.0, math.pi)                    # its face at x = 0, facing +x

  def stand_routine(self):
    self._did("stand")
    yield from ()

  def drive_to_routine(self, x, y, timeout, stop=None):
    self.pose = (x, y, math.pi)
    self._did("walk")
    return True
    yield

  def _face_board_routine(self, name):
    self._did("face")
    return self.faces.pop(0) if self.faces else True
    yield

  def _board_walk_in_routine(self, name):
    from pluggybot.legs.draw import STAND_M
    self.pose = (STAND_M, self.across, math.pi)
    self._did("walk in")
    return "stopped"
    yield

  def _drive_routine(self, seconds, v, w):
    yield from ()

  def _back_out_by_routine(self, *args):
    self._did("back out")
    yield from ()

  def rest_routine(self):
    self._did("lie")
    yield from ()

  def _board_estimate(self, name):
    return dw.BoardEstimate(x_face=0.6, y_mid=0.0, z_mid=0.2)

  def _off_the_board_routine(self, plotter):
    self._did("off the board")
    yield from ()


class _Plotter:
  """The plotter as the walk meets it: its probes recorded, never a touch."""

  def __init__(self, mission, board, on_stroke=None):
    self.mission, self.should_stop = mission, None

  def calibrate_routine(self, est):
    self.mission._did("probe")
    return {"ok": False}
    yield

  def fell(self) -> bool:
    return False


def test_a_drawing_lies_down_only_where_it_can_draw(monkeypatch):
  """A fall on the walk in threw the pen, and the body got up: it neither
  lies down nor probes the board with an empty fork (the posture alone
  said nothing had happened, and the pen on the floor answered its quill);
  a re-face that lost the board leaves it at the look point, where every
  probe is out of reach; a fall on the way to the board ends it there,
  before it faces the board."""
  monkeypatch.setattr(dw, "PenPlotter", _Plotter)
  square = strokes.program("square", size=0.03)
  for walk, why in ((_Walk(fall_on="walk in"), "fell"),
                    (_Walk(across=0.05, faces=(True, False)), "lost"),
                    (_Walk(fall_on="walk"), "fell")):
    rec = tick.run(SimpleNamespace(step=lambda cmd: None),
                   walk.draw_routine("whiteboard_a", None, square, patience=600.0))
    assert rec["why"] == why and not rec["drew"], (walk.did, rec)
    assert "lie" not in walk.did and "probe" not in walk.did, walk.did
  assert walk.did == ["stand", "walk"], walk.did
  ok = _Walk()
  tick.run(SimpleNamespace(step=lambda cmd: None),
           ok.draw_routine("whiteboard_a", None, square, patience=600.0))
  assert ok.did[-5:] == ["lie", "probe", "off the board", "stand", "back out"], \
      "the premise: lined up and holding the pen, it lies down and probes"


# ---- the jobs, on the stub body ---------------------------------------------------------


def _drawing_life(**kw):
  from pluggybot.lifecycle import QUAD_HOME, board_book, points_ledger
  from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path
  return stub_life(QUAD_HOME, boards=board_book(QUAD_HOME), ledger=points_ledger(None), **kw)


def test_each_board_job_finds_the_board_fetches_the_pen_draws_and_hangs_it_back():
  """The three jobs' errands (#406): the board found by its left tag round
  its house's ADDRESS -- nothing finer -- BEFORE the pen is fetched (a body
  carrying turns slowly), the figure drawn, the pen hung back; an answer's
  figure is the digits its claim froze, and a claim with no clean answer
  builds nothing."""
  from pluggybot.economy.tasks import KINDS, TaskBoard
  from pluggybot.home.places import area
  from pluggybot.lifecycle import DRAW_FIND_PATIENCE_S, QUAD_HOME, errand_for_task
  board = TaskBoard()
  at = area("whiteboard_b")["address"]
  for kind, params, said, figure in (
      ("draw_figure", {"program": "tree"}, "", "tree"),
      ("rate_artwork", {"program": "robot"}, "", "robot"),
      ("whiteboard_answer", {"question": "What is 2 + 3?"}, "5", "answer:5")):
    task = board.offer(kind, "whiteboard_b", params=params,
                       secret={"answer": "5"} if said else None)
    errand = errand_for_task(task, QUAD_HOME, answer=said)
    steps = errand.program.steps()
    assert [s.verb for s in steps] == ["find", "fetch", "draw", "stow"], kind
    assert steps[0].args == {"tag": 40, "x": at["x"], "y": at["y"],
                             "patience": DRAW_FIND_PATIENCE_S}
    assert steps[1].args == {"tool": "module_pen"}
    assert steps[2].args == {"board": "whiteboard_b", "figure": figure}
    assert errand.task == KINDS[kind].task and errand.detail["board"] == "whiteboard_b"
  task = board.offer("whiteboard_answer", "whiteboard_a", params={"question": "What is 2 + 3?"},
                     secret={"answer": "5"})
  assert errand_for_task(task, QUAD_HOME, answer="8.0") is None


def test_a_procedure_may_draw_a_figure_and_never_an_answer():
  """`draw` is a legs verb where the world has boards, a rack and places;
  an `answer:<digits>` figure is the question's errand's alone -- its own
  facts carry it, and no procedure the robot writes may name one."""
  from pluggybot.lifecycle import QUAD_HOME, world_facts
  from pluggybot.procedure import steps as st
  facts = world_facts(QUAD_HOME)
  assert "draw" in facts.verbs
  ok = {"board": "whiteboard_a", "figure": "house"}
  assert st.check_step(st.VERBS["draw"], ok, facts) == []
  answer = {"board": "whiteboard_a", "figure": "answer:42"}
  assert st.check_step(st.VERBS["draw"], answer, facts)
  assert st.check_step(st.VERBS["draw"], answer, world_facts(QUAD_HOME, answer="42")) == []
  assert st.check_step(st.VERBS["draw"], {**ok, "board": "board_c"}, facts)


def test_a_drawing_job_is_paid_for_the_ink_on_the_board_and_a_board_never_found_says_so():
  """On the stub: the board found (the test saw its tags in), the stub's
  pen inks the figure exactly, and the job is paid off the BOARD BOOK --
  the board erased once, every line this robot's. A board never found
  leads its verdict with why (#350)."""
  from pluggybot.lifecycle import QUAD_HOME, draw_errand, figure_program
  life = _drawing_life()
  try:
    life.body.draws = True
    life.body.places.see(38, -1.98, 0.735, 0.0, 0.0)
    life.body.places.see(39, -1.98, 1.265, 0.0, 0.0)
    result = life.run_errand(draw_errand(QUAD_HOME, "whiteboard_a", "house"))
    assert result["verdict"]["ok"] and result["points"] > 0, result["verdict"]
    rec = life.boards["whiteboard_a"]
    house = figure_program(QUAD_HOME, "whiteboard_a", "house")
    assert rec.clears == 1 and rec.strokes == len(house.strokes)
    assert {line["by"] for line in rec.lines} == {life.root}
    assert life.body.drew == [("whiteboard_a", "house")]
    missed = life.run_errand(draw_errand(QUAD_HOME, "whiteboard_b", "house"))
    assert not missed["verdict"]["ok"]
    assert missed["verdict"]["reason"].startswith("never found whiteboard_b: did not find tag 40")
    assert life.body.drew == [("whiteboard_a", "house")], "nothing fetched, nothing drawn"
  finally:
    life.body.close()


def test_the_board_is_erased_by_the_first_ink_never_by_a_stroke_that_missed():
  """The rover's #30, kept: the erase rides the first stroke that INKED. A
  pen that never touched leaves yesterday's drawing up, as it is."""
  life = _drawing_life()
  try:
    hook = life.ink_hook("whiteboard_a", "house")
    hook(0, [], "house")
    assert life.boards["whiteboard_a"].clears == 0
    hook(1, [(0.0, 0.0), (0.01, 0.0)], "house")
    hook(2, [(0.0, 0.01), (0.01, 0.01)], "house")
    assert life.boards["whiteboard_a"].clears == 1 and life.boards["whiteboard_a"].strokes == 2
    assert life.ink_hook("board_c", "house") is None
  finally:
    life.body.close()


def test_an_answer_is_paid_only_when_it_is_right_and_on_the_board():
  """`whiteboard_answer` on the stub, through the claim the mind makes: the
  right answer, drawn, is paid off the board's ink against the glyphs; a
  wrong one is drawn and paid nothing, and its line says wrong without
  saying what was right."""
  from pluggybot.economy.tasks import TaskBoard
  life = _drawing_life(tasks=TaskBoard())
  try:
    life.body.draws = True
    life.body.places.see(38, -1.98, 0.735, 0.0, 0.0)
    life.body.places.see(39, -1.98, 1.265, 0.0, 0.0)
    for said, ok in (("42", True), ("41", False)):
      task = life.tasks.offer("whiteboard_answer", "whiteboard_a",
                              params={"question": "What is six times seven?"},
                              secret={"answer": "42"}, t=float(life.data.time))
      assert life._claim_task(task.id, answer=said)
      result = life.run_errand(life.errands.pop(0))
      verdict = result["verdict"]
      assert verdict["ok"] is ok, verdict
      assert ("correct" if ok else "wrong") in verdict["reason"]
      assert "42" not in verdict["reason"] or ok
  finally:
    life.body.close()


def test_a_job_that_inks_nothing_is_never_paid_for_the_ink_already_up():
  """Only the first ink erases a board, so after a paid answer the board
  still carries it: an answer, a figure and an artwork whose pen inks
  nothing on it are never paid for that ink. The board is read BEFORE a
  program's errand as it was before the rover's native one
  (`scoring.board_before`); with the cage's reading alone, the second
  answer was paid "correct, 0.0 mm from the glyphs"."""
  from pluggybot.economy.tasks import TaskBoard
  life = _drawing_life(tasks=TaskBoard())
  try:
    life.body.places.see(38, -1.98, 0.735, 0.0, 0.0)
    life.body.places.see(39, -1.98, 1.265, 0.0, 0.0)
    question = {"question": "What is six times seven?"}
    for kind, params, said, draws in (
        ("whiteboard_answer", question, "42", True),
        ("whiteboard_answer", question, "42", False),
        ("draw_figure", {"program": "house"}, "", False),
        ("rate_artwork", {"program": "robot"}, "", False)):
      life.body.draws = draws
      task = life.tasks.offer(kind, "whiteboard_a", params=params,
                              secret={"answer": "42"} if said else None,
                              t=float(life.data.time))
      assert life._claim_task(task.id, answer=said)
      result = life.run_errand(life.errands.pop(0))
      assert result["verdict"]["ok"] is draws, (kind, result["verdict"])
      assert (result["points"] > 0) is draws, (kind, result["verdict"])
    assert life.boards["whiteboard_a"].clears == 1, "the answer is still up"
  finally:
    life.body.close()


def test_a_walks_first_turn_to_face_its_route_is_no_stagnation(monkeypatch):
  """After a drawing the rack is behind the robot, and carrying the pen it
  turns at most `W_CARRY`: the half-turn took 9 of the 10 s a walk may go
  without progress, and the stow gave up "stalled" as it finished turning.
  A walk's first turn to face its route is no stagnation -- a turn of 12.6 s
  here; and without the rule (aimed from the start) the same walk stalls.
  ...for `STAGNATION_S` at most: a body that never turns stalls, never
  spending its whole patience on the turn."""
  from pluggybot import navigator as nav
  from test_make_way import GOAL, STEPPER, _Drive  # noqa: I001 -- tests/ is on sys.path

  class Turning(_Drive):
    def _nav_routine(self, v, w):
      self.data.time += 0.1
      x, y, th = self.pose
      err = nav.wrap_angle(math.atan2(GOAL[1] - y, GOAL[0] - x) - th)
      if abs(err) > math.radians(5.0):
        self.pose = (x, y, th + math.copysign(min(0.025, abs(err)), err))   # 0.25 rad/s
      else:
        self.pose = (GOAL[0], GOAL[1], th)
      yield v, w

  def walk(cls=Turning):
    drive = cls([], plans=[GOAL])
    drive.pose = (0.0, 0.0, math.pi)                # facing away from its route
    return tick.run(STEPPER, drive.drive_to_routine(*GOAL, 60.0)), drive.last_drive

  class Stuck(Turning):
    def _nav_routine(self, v, w):
      self.data.time += 0.1
      yield v, w

  arrived, rec = walk()
  assert arrived, rec
  arrived, rec = walk(Stuck)
  assert not arrived and rec["why"] == "stalled", rec
  assert rec["seconds"] <= 2 * nav.STAGNATION_S + 0.5, rec
  monkeypatch.setattr(nav, "AIMED_RAD", math.pi + 1.0)
  arrived, rec = walk()
  assert not arrived and rec["why"] == "stalled", "the premise: it gave up turning"
