"""Tools on the floor (issue #347): every verb that drives carries the tool
in its carrying pose, `draw` takes the route to its board, and a lost tool
goes home by itself.

Every rule here is pinned without flying a mission: a stubbed drive that
reads the setpoints at its first command, a fake clock and a fake
whereabouts for the lost-tool clock, a module put on the floor by its qpos.
"""

from types import SimpleNamespace

import mujoco
import pytest

from pluggybot import tick
from pluggybot.lifecycle import (AUTO_RESTART_BY, LOST_TOOL_S, HubLifecycle,
                                 world_config, world_facts)
from pluggybot.mission.mission import HubMission
from pluggybot.procedure import lang, steps as st
from pluggybot.rack.coupling import STATION_YS
from pluggybot.rack.swap import ARM_EXT
from pluggybot.tools.gripper import CARRY_LIFT, MODULE_DRIVE_LIFT

HUB = world_facts("room_hub")
#: Rowan's `pen_check`, 2026-09-24: the setpoints `draw` drove off with.
ROWAN = {"lift": 0.15, "arm": 0.10, "carriage": 0.03}


@pytest.fixture(scope="module")
def hub_model():
  return mujoco.MjModel.from_xml_path("models/hub_world.xml")


def _pen_life(model, monkeypatch):
  """A robot with the pen on its fork (as `_carried` reads it) and its
  setpoints where Rowan's procedure left them; `drive_to` records the
  setpoints at its first command and arrives."""
  data = mujoco.MjData(model)
  mission = HubMission(model, data, viewer=None, realtime=False)
  mission.start_at(0.5, 3.0, 0.0)
  acts = {"lift": mission.swap.lift_act, "arm": mission.swap.arm_act,
          "carriage": model.actuator("pen_carriage").id}
  for name, value in ROWAN.items():
    data.ctrl[acts[name]] = value
  seen: list = []

  def drive_to_routine(x, y, timeout=None):
    seen.append({name: float(data.ctrl[a]) for name, a in acts.items()})
    return tick.result(True)
  mission.drive_to_routine = drive_to_routine
  monkeypatch.setattr(st, "_carried", lambda life: "module_pen")
  life = SimpleNamespace(mission=mission, model=model, data=data, module="module_pen",
                         swaps_done=0, interrupted=lambda: False,
                         _say=lambda *a, **k: None, world="room_hub", boards=None,
                         ledger=None, battery=SimpleNamespace(fraction=0.5, energy_wh=1.0))
  return life, seen, acts


def _by_language(life, src):
  return life.mission.run(lang.run_procedure_routine(
    life, lang.compile_procedure(src, HUB), HUB))


def _by_program(life, steps):
  return life.mission.run(st.run_program_routine(
    life, st.Program.single("go", [st.Step(v, a) for v, a in steps]), HUB))


# ---- 1. every verb that drives carries the tool in its carrying pose ---------


@pytest.mark.parametrize("runner", ["language", "program"])
def test_a_drive_starts_with_the_pen_in_its_carrying_pose(hub_model, monkeypatch, runner):
  """Rowan's `pen_check` moved the lift, the arm and the carriage, and the
  next verb drove off like that: the setpoints are the carrying pose
  before the drive's FIRST command, whichever runner ran the verb."""
  life, seen, _ = _pen_life(hub_model, monkeypatch)
  r = (_by_language(life, "def go():\n  drive_to(1.0, 1.0)\n") if runner == "language"
       else _by_program(life, [("drive_to", {"x": 1.0, "y": 1.0})]))
  assert r["ok"], r
  assert seen[0] == pytest.approx({"lift": MODULE_DRIVE_LIFT, "arm": 0.0, "carriage": 0.0},
                                  abs=1e-6)


def test_a_verb_that_does_not_drive_leaves_the_pose_alone(hub_model, monkeypatch):
  """Posing a tool to USE it is the point of `move`: only a verb that
  moves the base is preceded by the carrying pose."""
  life, _, acts = _pen_life(hub_model, monkeypatch)
  r = _by_program(life, [("wait", {"seconds": 0.1}), ("look", {})])
  assert r["ok"], r
  assert {k: float(life.data.ctrl[a]) for k, a in acts.items()} == pytest.approx(ROWAN)
  assert {v for v, verb in st.VERBS.items() if verb.drives} == {
    "fetch", "stow", "drive_to", "drive", "face", "pick", "place", "draw"}


def test_the_mind_is_told_which_verbs_re_pose_the_tool():
  """A pose set with `move` is undone by the next verb that drives, and a
  procedure written without knowing it would read its own tool wrong: the
  rule names every driving verb, off the flags themselves."""
  from pluggybot.mind.overseer import procedure_rule
  drivers = [name for name, v in st.VERBS.items() if v.drives]
  rule = procedure_rule()
  assert f"moves the robot ({', '.join(f'`{d}`' for d in drivers)}) first" in rule
  assert "carrying pose" in rule


def test_a_tool_already_posed_costs_no_physics_step(hub_model, monkeypatch):
  """Only what a procedure moved is moved back: a verb that finds the pose
  already right steps nothing before its drive."""
  life, _, acts = _pen_life(hub_model, monkeypatch)
  life.data.ctrl[acts["lift"]] = MODULE_DRIVE_LIFT
  life.data.ctrl[acts["arm"]] = 0.0
  life.data.ctrl[acts["carriage"]] = 0.0
  t0 = float(life.data.time)
  life.mission.run(st.travel_pose_routine(life))
  assert float(life.data.time) == t0


def test_a_claw_holding_a_cube_keeps_it_and_its_carrying_height(hub_model, monkeypatch):
  """The return's pose SETS DOWN what the claw holds, which is right before
  a stow and wrong before `place`: a cube in the jaws travels where `pick`
  leaves it -- `CARRY_LIFT`, arm out -- and an empty claw tucks."""
  life, _, _ = _pen_life(hub_model, monkeypatch)
  claw = SimpleNamespace(held=lambda: "block_a_box")
  monkeypatch.setattr(st, "_claw", lambda life: claw)
  pose = {a: v for a, v, _ in st.travel_pose(life, "module_claw")}
  assert pose == {life.mission.swap.arm_act: ARM_EXT,
                  life.mission.swap.lift_act: CARRY_LIFT}
  claw.held = lambda: None
  pose = {a: v for a, v, _ in st.travel_pose(life, "module_claw")}
  assert pose == {life.mission.swap.arm_act: 0.0,
                  life.mission.swap.lift_act: MODULE_DRIVE_LIFT}


def test_draw_takes_the_route_to_its_board_before_its_own_approach(monkeypatch):
  """The drawing's use-phase opens with a straight line at the board, no
  planner (`drive_to_board_routine`), meant to settle from `use_at`. Called
  from the rack it drove Rowan through the house: every live `pen_check`
  that reached `draw` was knocked over, in pose or not (MEASURED on a
  local flight too: 94 deg, 14 s after the draw, the pen 3.7 m from its
  bay). The route comes first, and a route that fails draws nothing."""
  from pluggybot import lifecycle as lc
  order: list = []
  use_at: list = []
  real = lc.draw_errand_for

  def draw_errand_for(*a, **kw):
    errand = real(*a, **kw)
    use_at.append(errand.use_at)

    def use(life):
      order.append("use")
      return {"drew": True}
      yield
    errand.use = use
    return errand
  monkeypatch.setattr(lc, "draw_errand_for", draw_errand_for)
  monkeypatch.setattr(st, "_carried", lambda life: "module_pen")
  arrive = {"ok": True}

  def drive_to_routine(x, y, timeout=None):
    order.append(("drive_to", round(x, 3), round(y, 3)))
    return tick.result(arrive["ok"])
  life = SimpleNamespace(world="home", boards=lc.board_book("home"),
                         mission=SimpleNamespace(drive_to_routine=drive_to_routine,
                                                 pose=(0.5, -1.4, 1.57)))
  stepper = SimpleNamespace(_step_once=lambda *a: None)
  verdict = tick.run(stepper, st._draw(life, {"figure": "circle", "board": "whiteboard_b"}))
  sx, sy = use_at[0]            # where the native errand's carry drive goes
  assert order == [("drive_to", round(sx, 3), round(sy, 3)), "use"]
  assert verdict["ok"]
  order.clear()
  arrive["ok"] = False
  verdict = tick.run(stepper, st._draw(life, {"figure": "circle", "board": "whiteboard_b"}))
  assert order == [("drive_to", round(sx, 3), round(sy, 3))], "a failed route draws nothing"
  assert not verdict["ok"] and verdict["reason"].startswith("never reached whiteboard_b")
  assert verdict["used"] == {"error": "never reached the use pose"}


# ---- 2 + 3. a lost tool goes home, and never one in use -----------------------


def _life(**kw):
  cfg = world_config("room_hub")
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return HubLifecycle(model, data, realtime=False, world="room_hub", errand=False,
                      battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"], **kw)


@pytest.fixture
def clocked(monkeypatch):
  """A lifecycle with the clock on, a fake whereabouts per module (`bay`
  unless set), the return and History recorded, and one peer."""
  life = _life(lost_tool_after_s=LOST_TOOL_S)
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
  """`experiment.py` passes nothing: a measured run keeps its tools where
  they fell, on `restart_after_s`' terms."""
  life = _life()
  assert life.lost_tool_after_s is None
  monkeypatch.setattr(life, "tool_whereabouts", lambda m: "lost")
  monkeypatch.setattr(life, "_return_module", lambda m: pytest.fail("returned"))
  for t in (0.0, 1000.0):
    life.data.time = t
    life._lost_tool_step()


def test_a_pen_on_the_floor_is_back_on_its_bay_after_five_minutes():
  """The real whereabouts and the real return: the pen put on the floor by
  its qpos reads `lost`, and 300 s later it hangs on bay C again."""
  life = _life(lost_tool_after_s=LOST_TOOL_S)
  m, d = life.model, life.data
  swap = life.mission.swap
  assert life.tool_whereabouts("module_pen") == "bay"
  assert swap.module_state("module_pen")["bay"] == st.TOOL_BAYS["module_pen"]
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
  st_pen = swap.module_state("module_pen")
  assert st_pen["hung"] and st_pen["bay"] == st.TOOL_BAYS["module_pen"]


def test_whereabouts_reads_every_robot_and_its_own_bay(monkeypatch):
  """`fork` is ANY robot's fork, `swap` any robot working at the module's
  bay, and a module hung one bay over is `lost` -- nothing fetches it from
  there."""
  life = _life()
  pen_bay = STATION_YS[st.TOOL_BAYS["module_pen"]]
  home = life.mission.swap.module_state("module_pen")
  peer_state = {"on_fork": False}
  peer = SimpleNamespace(mission=SimpleNamespace(
    swapping_at=None,
    swap=SimpleNamespace(module_state=lambda m: {**home, **peer_state})))
  life.peers = [peer]
  assert life.tool_whereabouts("module_pen") == "bay"
  peer.mission.swapping_at = pen_bay
  assert life.tool_whereabouts("module_pen") == "swap"
  peer.mission.swapping_at = STATION_YS[st.TOOL_BAYS["module_lcd"]]
  assert life.tool_whereabouts("module_pen") == "bay", "another bay's swap"
  peer_state["on_fork"] = True
  assert life.tool_whereabouts("module_pen") == "fork"
  peer_state["on_fork"] = False
  real = life.mission.swap.module_state
  monkeypatch.setattr(life.mission.swap, "module_state",
                      lambda m: {**real(m), "bay": st.TOOL_BAYS["module_claw"]})
  assert life.tool_whereabouts("module_pen") == "lost"


def test_a_swap_holds_its_bay_for_as_long_as_it_runs(monkeypatch):
  """`swapping_at` is set from the swap's first step to its verdict, and
  cleared however it ends -- a stale one would keep a lost tool's clock
  stopped for ever."""
  life = _life()
  m = life.mission
  seen = []

  def body(station_y, verb, module=None, tries=2):
    seen.append(m.swapping_at)
    yield 0.0, 0.0
    if verb == "boom":
      raise RuntimeError("boom")
    return "ok"
  monkeypatch.setattr(m, "_swap_routine", body)
  assert m.run(m.swap_at_bay_routine(STATION_YS[2], "pick")) == "ok"
  assert seen == [STATION_YS[2]] and m.swapping_at is None
  with pytest.raises(RuntimeError):
    m.run(m.swap_at_bay_routine(STATION_YS[3], "boom"))
  assert m.swapping_at is None


def test_a_pair_has_one_hand():
  """The clock is the world's: the first robot's seam ticks it, or a tool
  would be put back twice."""
  from pluggybot.pair import build_pair
  lives = build_pair("room_hub", lost_tool_after_s=LOST_TOOL_S)
  assert [life.lost_tool_after_s for life in lives] == [LOST_TOOL_S, None]
