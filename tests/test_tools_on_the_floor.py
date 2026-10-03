"""Tools on the floor (issue #347): every verb that drives carries the tool
in its carrying pose, and a lost tool goes home by itself.

Every rule here is pinned without flying a mission: a stubbed walk that
reads the arm's setpoints at its first command, a fake clock and a fake
whereabouts for the lost-tool clock, a module put on the floor by its qpos.
The carrying pose and the rack are the served quadruped's; the clock is the
loop's bookkeeping, on the stub body (`tests/test_body.py`).
"""

from types import SimpleNamespace

import mujoco
import pytest

from pluggybot import tick
from pluggybot.legs import body as qb
from pluggybot.legs.arm import CARRY_Q
from pluggybot.lifecycle import (AUTO_RESTART_BY, LOST_TOOL_S, QUAD_HOME, HubLifecycle,
                                 world_config, world_facts)
from pluggybot.procedure import lang, steps as st
from pluggybot.rack.coupling import STATION_YS
from pluggybot.robot import world_spec
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path

FACTS = world_facts(QUAD_HOME)


@pytest.fixture(scope="module")
def quad_model():
  return world_spec(world_config(QUAD_HOME)["model"]).compile()


def _pen_life(model, monkeypatch):
  """A quadruped with the pen on its fork (as `_carried` reads it) and its
  arm at its stow, where a walk with nothing on the fork leaves it; its
  walk records the arm's setpoints at its first command and arrives."""
  body = qb.QuadBody(model, mujoco.MjData(model), realtime=False,
                     grid_bounds=world_config(QUAD_HOME)["grid_bounds"])
  body.start_at(1.5, 0.5, 0.0)
  acts = {"shoulder": body.actuator("arm_shoulder"), "elbow": body.actuator("arm_elbow")}
  seen: list = []

  def drive_to_routine(x, y, timeout=None, stop=None):
    seen.append({name: body.setpoint(a) for name, a in acts.items()})
    return tick.result(True)
  body.mission.drive_to_routine = drive_to_routine
  monkeypatch.setattr(st, "_carried", lambda life: "module_pen")
  life = SimpleNamespace(body=body, model=model, data=body.data, module="module_pen",
                         swaps_done=0, interrupted=lambda: False,
                         _say=lambda *a, **k: None, world=QUAD_HOME, boards=None,
                         ledger=None, drive_why=lambda x, y: "the walk gave up (why)",
                         battery=SimpleNamespace(fraction=0.5, energy_wh=1.0))
  return life, seen, acts


def _by_language(life, src):
  return life.body.run(lang.run_procedure_routine(
    life, lang.compile_procedure(src, FACTS), FACTS))


def _by_program(life, steps):
  return life.body.run(st.run_program_routine(
    life, st.Program.single("go", [st.Step(v, a) for v, a in steps]), FACTS))


# ---- 1. every verb that drives carries the tool in its carrying pose ---------


@pytest.mark.parametrize("runner", ["language", "program"])
def test_a_walk_starts_with_the_tool_in_its_carrying_pose(quad_model, monkeypatch, runner):
  """A procedure leaves the arm wherever it moved it, and the next verb
  walked off like that: the setpoints are the carrying pose before the
  walk's FIRST command, whichever runner ran the verb."""
  life, seen, _ = _pen_life(quad_model, monkeypatch)
  try:
    r = (_by_language(life, "def go():\n  drive_to(1.0, 1.0)\n") if runner == "language"
         else _by_program(life, [("drive_to", {"x": 1.0, "y": 1.0})]))
    assert r["ok"], r
    assert seen[0] == pytest.approx({"shoulder": CARRY_Q[0], "elbow": CARRY_Q[1]},
                                    abs=1e-6)
  finally:
    life.body.close()


def test_a_verb_that_does_not_drive_leaves_the_pose_alone(quad_model, monkeypatch):
  """Posing a tool to USE it is the point of `move`: only a verb that
  moves the base is preceded by the carrying pose."""
  life, _, acts = _pen_life(quad_model, monkeypatch)
  try:
    before = {k: life.body.setpoint(a) for k, a in acts.items()}
    r = _by_program(life, [("wait", {"seconds": 0.1})])
    assert r["ok"], r
    assert {k: life.body.setpoint(a) for k, a in acts.items()} == pytest.approx(before)
    assert before != pytest.approx({"shoulder": CARRY_Q[0], "elbow": CARRY_Q[1]})
  finally:
    life.body.close()
  assert {v for v, verb in st.VERBS.items() if verb.drives} == {
    "fetch", "stow", "drive_to", "drive", "face", "find", "press", "draw", "pick",
    "place", "survey"}


def test_the_mind_is_told_which_verbs_re_pose_the_tool():
  """A pose set with `move` is undone by the next verb that drives, and a
  procedure written without knowing it would read its own tool wrong: the
  rule names every driving verb it can run, off the flags themselves."""
  from pluggybot.mind.overseer import procedure_rule
  verbs = st.BODY_VERBS + st.SWAP_VERBS + st.PLACE_VERBS + st.PLATE_VERBS
  drivers = [name for name in verbs if st.VERBS[name].drives]
  rule = procedure_rule(swaps=True, places=True, plates=True)
  assert f"({', '.join(f'`{d}`' for d in drivers)}) first" in rule
  assert "carrying pose" in rule


def test_a_tool_already_posed_costs_no_physics_step(quad_model, monkeypatch):
  """Only what a procedure moved is moved back: a verb that finds the pose
  already right steps nothing before its walk."""
  life, _, _ = _pen_life(quad_model, monkeypatch)
  try:
    monkeypatch.setattr(st, "_carried", lambda life: None)     # folded is right
    t0 = float(life.data.time)
    life.body.run(st.travel_pose_routine(life))
    assert float(life.data.time) == t0
  finally:
    life.body.close()


# ---- 2 + 3. a lost tool goes home, and never one in use -----------------------


def _life(**kw):
  """The clock's bookkeeping, on the stub."""
  return stub_life(**kw)


def _quad(**kw):
  """A world with the tools on their rack: the quadruped's house."""
  cfg = world_config(QUAD_HOME)
  spec = world_spec(cfg["model"])
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return HubLifecycle(model, data, realtime=False, world=QUAD_HOME, spec=spec,
                      battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"], **kw)


@pytest.fixture
def clocked(monkeypatch):
  """A lifecycle with the clock on, a fake whereabouts per module (`bay`
  unless set), the return and History recorded, and one peer. In a world
  that has the modules: the clock skips one the world no longer has."""
  life = _quad(lost_tool_after_s=LOST_TOOL_S)
  where: dict = {}
  returned, mine, theirs, events = [], [], [], []
  monkeypatch.setattr(life, "tool_whereabouts", lambda m: where.get(m, "bay"))
  monkeypatch.setattr(life, "_return_module", returned.append)
  monkeypatch.setattr(life, "_remember", mine.append)
  life.peers = [SimpleNamespace(_remember=theirs.append)]
  life.on_event.append(events.append)

  def at(t):
    life.data.time = t
    life._lost_tool_step()
  return SimpleNamespace(life=life, where=where, returned=returned, mine=mine,
                         theirs=theirs, events=events, at=at)


def test_a_lost_tool_goes_home_at_five_minutes_and_not_before(clocked):
  c = clocked
  c.where["module_pen"] = "lost"
  for t in (0.0, 100.0, 299.0):
    c.at(t)
  assert c.returned == []
  c.at(300.0)
  assert c.returned == ["module_pen"]
  c.where["module_pen"] = "bay"
  c.at(301.0)
  c.at(700.0)
  assert c.returned == ["module_pen"], "once, and the clock stops when it is home"


def test_a_return_is_the_world_s_hand_and_never_an_intervention(clocked):
  """On the auto stand-up's terms: a `reset_tool` by `auto-restart`, in
  both robots' History, and nothing in `interventions`."""
  c = clocked
  c.where["module_pen"] = "lost"
  c.at(0.0)
  c.at(300.0)
  line = "module_pen lay on the floor for 5 minutes and was put back on its bay"
  assert c.mine == [line] and c.theirs == [line]
  [event] = c.events
  assert event == {"type": "reset_tool", "t": 300.0, "robot": "pluggybot",
                   "module": "module_pen", "by": AUTO_RESTART_BY, "auto": True,
                   "intervention": False, "lostS": 300.0, "detail": line}
  assert c.life.tools_returned == [event]
  assert c.life.interventions == []


@pytest.mark.parametrize("busy", ["fork", "swap", "bay"])
def test_anything_but_lost_restarts_the_clock(clocked, busy):
  """On a fork (any robot's, alive or dead), mid-swap, or back on its bay
  for one look: the five minutes start again from the next time it is
  seen lost."""
  c = clocked
  c.where["module_pen"] = "lost"
  c.at(0.0)
  c.at(200.0)
  c.where["module_pen"] = busy
  c.at(250.0)
  c.where["module_pen"] = "lost"
  for t in (251.0, 300.0, 549.0):
    c.at(t)
  assert c.returned == []
  c.at(551.0)
  assert c.returned == ["module_pen"]


def test_a_tool_in_use_is_never_returned(clocked):
  c = clocked
  for busy in ("fork", "swap"):
    c.where["module_claw"] = busy
    for t in (0.0, 300.0, 1000.0, 5000.0):
      c.at(t)
  assert c.returned == []


def test_the_clock_is_off_unless_it_is_asked_for(monkeypatch):
  """A lifecycle nobody asked for it keeps its tools where they fell, on
  `restart_after_s`' terms: ON in `serve.py` only."""
  life = _quad()
  assert life.lost_tool_after_s is None
  monkeypatch.setattr(life, "tool_whereabouts", lambda m: "lost")
  monkeypatch.setattr(life, "_return_module", lambda m: pytest.fail("returned"))
  for t in (0.0, 1000.0):
    life.data.time = t
    life._lost_tool_step()


def test_a_pen_on_the_floor_is_back_on_its_bay_after_five_minutes():
  """The real whereabouts and the real return: the pen put on the floor by
  its qpos reads `lost`, and 300 s later it hangs on its own bay again."""
  life = _quad(lost_tool_after_s=LOST_TOOL_S)
  m, d = life.model, life.data
  assert life.tool_whereabouts("module_pen") == "bay"
  assert life.body.module_state("module_pen")["bay"] == st.TOOL_BAYS["module_pen"]
  qadr = int(m.jnt_qposadr[m.body("module_pen").jntadr[0]])
  d.qpos[qadr:qadr + 3] = (d.qpos[qadr] + 1.0, d.qpos[qadr + 1] + 1.0, 0.03)
  mujoco.mj_forward(m, d)
  assert life.tool_whereabouts("module_pen") == "lost"
  for t in (10.0, 309.0):
    d.time = t
    life._lost_tool_step()
  assert life.tool_whereabouts("module_pen") == "lost"
  d.time = 310.0
  life._lost_tool_step()
  st_pen = life.body.module_state("module_pen")
  assert st_pen["hung"] and st_pen["bay"] == st.TOOL_BAYS["module_pen"]


def _qadr(model, body):
  return int(model.jnt_qposadr[model.body(body).jntadr[0]])


def test_a_lost_tool_is_never_put_into_a_taken_bay():
  """`_return_module` writes the pose: into a bay another module hangs in,
  the two would be one body inside the other. The pen lost at t=0 waits
  while the LCD sits in its bay (lost itself, one bay over, from t=100);
  at t=400 the LCD goes home first and the pen follows, both hung."""
  life = _quad(lost_tool_after_s=LOST_TOOL_S)
  m, d = life.model, life.data
  pen, lcd = _qadr(m, "module_pen"), _qadr(m, "module_lcd")
  d.qpos[pen:pen + 3] = (d.qpos[pen] + 1.0, d.qpos[pen + 1] + 1.0, 0.03)
  mujoco.mj_forward(m, d)

  def at(t):
    d.time = t
    life._lost_tool_step()
  at(0.0)
  d.qpos[lcd:lcd + 7] = m.qpos0[pen:pen + 7]          # the LCD hangs in the pen's
  mujoco.mj_forward(m, d)
  for t in (100.0, 300.0, 399.0):
    at(t)
  assert life.tools_returned == [], "never into a taken bay"
  assert life.body.module_state("module_lcd")["bay"] == st.TOOL_BAYS["module_pen"]
  at(400.0)
  assert [e["module"] for e in life.tools_returned] == ["module_lcd", "module_pen"]
  for module in ("module_lcd", "module_pen"):
    here = life.body.module_state(module)
    assert here["hung"] and here["bay"] == st.TOOL_BAYS[module], module


def test_a_retired_tool_leaves_no_clock_behind(clocked):
  c = clocked
  c.life._lost_since["module_gone"] = 0.0
  c.at(0.0)
  assert "module_gone" not in c.life._lost_since


def test_whereabouts_reads_every_robot_and_its_own_bay(monkeypatch):
  """`fork` is ANY robot's fork, `swap` any robot working at the module's
  bay, and a module hung one bay over is `lost` -- nothing fetches it from
  there."""
  life = _life()
  pen_bay = STATION_YS[st.TOOL_BAYS["module_pen"]]
  home = life.body.module_state("module_pen")
  peer_state = {"on_fork": False}
  peer = SimpleNamespace(body=SimpleNamespace(
    swapping_at=None, module_state=lambda m: {**home, **peer_state}))
  life.peers = [peer]
  assert life.tool_whereabouts("module_pen") == "bay"
  peer.body.swapping_at = pen_bay + 1e-9           # a station, however computed
  assert life.tool_whereabouts("module_pen") == "swap"
  peer.body.swapping_at = STATION_YS[st.TOOL_BAYS["module_lcd"]]
  assert life.tool_whereabouts("module_pen") == "bay", "another bay's swap"
  peer_state["on_fork"] = True
  assert life.tool_whereabouts("module_pen") == "fork"
  peer_state["on_fork"] = False
  real = life.body.module_state
  monkeypatch.setattr(life.body, "module_state",
                      lambda m: {**real(m), "bay": st.TOOL_BAYS["module_claw"]})
  assert life.tool_whereabouts("module_pen") == "lost"


def test_a_swap_holds_its_bay_for_as_long_as_it_runs(quad_model):
  """`swapping_at` is set from the swap's first step at the bay to its
  verdict, and cleared however it ends -- a stale one would keep a lost
  tool's clock stopped for ever."""
  body = qb.QuadBody(quad_model, mujoco.MjData(quad_model), realtime=False,
                     grid_bounds=world_config(QUAD_HOME)["grid_bounds"])
  body.start_at(1.5, 0.5, 0.0)
  try:
    m = body.mission
    seen = []

    def at_the_bay(bay, module, rec):
      seen.append(m.swapping_at)
      yield qb.STILL
      if module == "boom":
        raise RuntimeError("boom")
      return "arrived"
    m._to_the_bay_routine = lambda bay, rec: tick.result("ok")
    m._fetch_at_routine = at_the_bay
    assert body.run(m.fetch_routine(2, "module_claw")) == "arrived"
    assert seen == [STATION_YS[2]] and m.swapping_at is None
    with pytest.raises(RuntimeError):
      body.run(m.fetch_routine(1, "boom"))
    assert m.swapping_at is None
  finally:
    body.close()


def test_a_pair_has_one_hand():
  """The clock is the world's: the first robot's seam ticks it, or a tool
  would be put back twice."""
  from pluggybot.pair import build_pair
  lives = build_pair(QUAD_HOME, lost_tool_after_s=LOST_TOOL_S)
  assert [life.lost_tool_after_s for life in lives] == [LOST_TOOL_S, None]
