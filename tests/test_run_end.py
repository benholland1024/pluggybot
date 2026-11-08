"""A run ends on its hour, even mid-charge (issue #654).

The served process restarts after each hour of sim time and the world
carries on from it (`Keeper`, `HubLifecycle.continuing`). A charge never
looked at that hour: run to full, up to 85 minutes, the other robot stood
`DONE` until it finished. So a charge stops at the run's end with the plug
in, the pack filling on while the next process's robot decides, and
everything that assumed a robot only stands plugged in inside `charge_routine`
reads the plug now. A world that will not carry on finishes its charge as
it always did (`test_a_day_runs_on_the_stub_charge_and_all`).

The loop's bookkeeping, on the stub: its plug is `StubBody.on_charger`.
"""
import math
from types import SimpleNamespace

import pytest

from pluggybot import lifecycle as lc
from pluggybot import tick
from pluggybot.activity import encounter as enc
from pluggybot.body import RackPose, StubBody
from pluggybot.economy import energy
from pluggybot.mind import overseer as ov
from pluggybot.mission.errand import programmed_errand
from pluggybot.procedure.steps import Program, Step
from pluggybot.robot import SECOND

from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path
from test_continuation import _history, _prelude, _saved, _stub
from test_hand_jobs import _life as _counter_life
from test_hand_jobs import _offer

HOUR_S = 120.0


class _Done(Exception):
  """Raised at the top of the day loop, the claim settled."""


def _peer(x: float, y: float, state: str = "IDLE"):
  """The other robot's public surface (`test_rack_contention._peer`)."""
  pos = [x, y]
  return SimpleNamespace(robot_name="Rowan", root=SECOND.root, state=state,
                         pos=pos, down=lambda: False,
                         body=SimpleNamespace(pose_xy=lambda: (pos[0], pos[1])))


def _reported(life, peer) -> None:
  def rule(x, y):
    d = math.hypot(peer.pos[0] - x, peer.pos[1] - y)
    return d if d < 0.45 else None
  life.body.peer_on_the_goal = rule


def _life(peer: bool = False, continuing: bool = True):
  """The stub with its charge logic round the humanoids' station (as
  `test_rack_contention._life`), its world carrying on from this run, and
  no charge paid for by the walk or the draw: what is read is the plug."""
  cfg = lc.world_config("home_humanoid")
  x, y, yaw = cfg["dock"]
  body = StubBody(rack=RackPose(x, y, yaw + math.pi), grid_bounds=cfg["grid_bounds"],
                  draw_w=0.0, charge_w=energy.load("home_humanoid").charge_w)
  life = stub_life(body=body)
  life.home_pose = tuple(float(v) for v in cfg["start"])
  life.continuing = continuing
  if peer:
    far = _peer(cfg["dock"][0] + 8.0, cfg["dock"][1])
    life.peers = [far]
    _reported(life, far)
  return life


def _at_the_station(life) -> None:
  """Standing on the seat, inside `RACK_CLEAR_M`: `_clear_rack_routine` walks
  anyone here off unless it reads the plug."""
  life.body.x, life.body.y = life.body.charge_seat()
  assert life.rack_distance(life.body.x, life.body.y) < lc.RACK_CLEAR_M


def _walks(life) -> list:
  """Every walk the body is asked for WITH ITS PLUG IN, answered at once."""
  drives: list = []

  def walk(*a, **kw):
    if life.body.on_charger:
      drives.append(a[:2])
    return tick.result(True)
  life.body.go_to_routine = walk
  return drives


def _plugged_in(life, frac: float = 0.4) -> None:
  life.body.on_charger = True
  life.battery.energy_wh = life.battery.capacity_wh * frac


# ---- the charge stops on the hour --------------------------------------------


@pytest.mark.parametrize("peer", [False, True], ids=["alone", "with-the-other"])
def test_a_charge_stops_at_the_runs_end_with_its_plug_in_and_the_day_ends(peer):
  """Rowan stood `DONE` for 20-70 minutes while Luca's charge ran to full,
  because a charge never looked at the run's end. The loop ends on the
  hour, what it got is banked, and the plug is left in: the day's end
  leaves the station for robots that have finished at it, and this one has
  not (`_leave_rack_routine` walked off with the plug in)."""
  life = _life(peer=peer)
  life.battery.energy_wh = life.low_battery_wh * 0.5   # the loop goes to charge
  _at_the_station(life)
  drives = _walks(life)
  summary = life.run(start=(0.5, 3.0, 0.0), max_sim_time=HOUR_S)
  assert float(life.data.time) == pytest.approx(HOUR_S, abs=1.0), "ended on its hour"
  assert life.body.on_charger, "its plug is still in"
  assert summary["charge_cycles"] == 0, "not a charge finished"
  [row] = [v for v in life.verdicts if v["task"] == "charge"]
  assert row["metrics"]["gainedWh"] > 0.0, "what it got is banked"
  assert not drives, f"the day's end walked it off with its plug in: {drives}"


def test_the_state_is_still_charge_so_the_next_run_can_say_it_was_cut():
  """`DONE` would be the word for a robot that is doing nothing, and the
  next run's first line goes by it (`_resumed_line`); a `DONE` robot
  plugged in is neither."""
  life = _life()
  life.battery.energy_wh = life.low_battery_wh * 0.5
  summary = life.run(start=(0.5, 3.0, 0.0), max_sim_time=HOUR_S)
  assert summary["state"] == "CHARGE"
  # ...and a robot not plugged in at its day's end is DONE, as it always was
  life = _life()
  life.battery.energy_wh = life.battery.capacity_wh
  assert life.run(start=(0.5, 3.0, 0.0), max_sim_time=1.0)["state"] == "DONE"


def test_a_charge_begun_past_the_hour_banks_no_row_of_nothing():
  """Plugged in as its time ran out, it never held a moment: a row of 0 Wh
  in 0 s says a charge failed that never began."""
  life = _life()
  life.max_sim_time = 0.0
  _plugged_in(life)
  life.body.run(life.charge_routine())
  assert life.body.on_charger and not life.verdicts


def test_a_charge_that_is_full_on_the_hour_is_finished_and_unplugged():
  """The hour stops a charge still filling, never one that is done: full,
  it backs off as it always did."""
  life = _life()
  life.max_sim_time = 0.0
  _plugged_in(life, frac=1.0)
  life.body.run(life.charge_routine())
  assert not life.body.on_charger and life.charge_cycles == 1


def test_a_world_that_will_not_carry_on_finishes_its_charge():
  """The cut is for a next run to carry on from. With none (`continuing`
  off: a demo, a test), the old day stands: the charge runs to full and
  unplugs, its budget or not."""
  life = _life(continuing=False)
  life.max_sim_time = 0.0
  _plugged_in(life, frac=0.895)
  life.body.run(life.charge_routine())
  assert not life.body.on_charger and life.charge_cycles == 1


# ---- what the next run does, plugged in ---------------------------------------


def test_a_charge_decided_while_plugged_in_goes_straight_on():
  """Restarted plugged in, a `charge` walked to the standoff with the plug
  in, pulled it out and docked again."""
  life = _life()
  life.max_sim_time = 1e9
  _plugged_in(life, frac=0.89)
  drives = _walks(life)
  docks: list = []

  def dock():
    docks.append(float(life.data.time))
    return "docked"
    yield
  life.body.dock_routine = dock
  t0 = float(life.data.time)
  assert life.body.run(life.go_charge_routine()) is True
  assert float(life.data.time) == t0 and not drives and not docks
  assert any("CHARGE: still plugged in" in ln for ln in life.log), life.log[-3:]


def test_a_decided_charge_while_plugged_in_fills_the_pack_and_only_then_unplugs():
  """The whole of the decided path: no walk, no dock, the charge on to full
  and its one undock at the end."""
  life = _life()
  life.max_sim_time = 1e9
  _plugged_in(life, frac=0.89)
  drives = _walks(life)
  assert life.body.run(life._after_decision_routine(ov.Decision(action="charge"))) == ""
  assert not drives and not life.body.on_charger
  assert life.charge_cycles == 1 and life.battery.fraction >= lc.CHARGED


def test_the_loop_with_no_mind_finishes_a_charge_it_wakes_plugged_in_to():
  """A rotation has no one to ask: it finishes what the last run's end cut
  short, rather than explore off with the plug in."""
  life = _life()
  _plugged_in(life, frac=0.89)
  drives = _walks(life)
  day = life.begin((0.5, 3.0, 0.0), max_sim_time=1e9)

  def stop():
    if life.charge_cycles:
      raise _Done
  life.at_loop_top.append(stop)
  with pytest.raises(_Done):
    life.body.run(day)
  assert not drives, f"it walked plugged in: {drives}"
  assert not life.body.on_charger and life.battery.fraction >= lc.CHARGED


def test_a_robot_plugged_in_stays_put_while_it_decides_or_stands_by():
  """The pair's loop clears the station before every decision, and at an
  `idle`: within `RACK_CLEAR_M` it walks home. Plugged in, that pulled the
  plug out before the mind was asked (so the charge it would have chosen
  was already gone)."""
  life = _life(peer=True)
  _plugged_in(life)
  _at_the_station(life)
  drives = _walks(life)
  life.body.run(life._leave_rack_routine())
  assert not drives, "walked off before it was asked"
  life.body.run(life._after_decision_routine(ov.Decision(action="idle")))
  assert not drives, "walked off to stand by"
  # ...and with no plug in, the rule that keeps a station clear still walks it
  life.body.on_charger = False
  gone: list = []
  life.body.go_to_routine = lambda *a, **kw: (gone.append(a[:2]), tick.result(True))[1]
  life.body.run(life._leave_rack_routine())
  assert gone[0] == life.home_pose[:2]


def test_anything_else_it_chooses_takes_its_plug_out_first():
  """A mind may choose not to go on charging, and walking off pulled the
  plug out past the port's 20 N. A hand takes it out first, as a charge's
  end does, before anything else the choice does; a `charge` never does."""
  life = _life()
  _plugged_in(life)
  order: list = []

  def undock():
    order.append("undock")
    life.body.on_charger = False
    return "arrived"
    yield
  life.body.undock_routine = undock

  def explore(**kw):
    order.append("explore")
    return "budget"
    yield
  life.explore_routine = explore
  life.body.run(life._after_decision_routine(ov.Decision(action="explore")))
  assert order == ["undock", "explore"]

  # ...a procedure too: the first thing it finds is the plug out
  order.clear()
  life.body.on_charger = True
  seen: list = []
  errand = programmed_errand(Program.single("lap", [Step("wait", {"seconds": 0.1})]),
                             task="program", name="procedure")
  real = life._run_program_routine

  def program(e):
    seen.append(life.body.on_charger)
    return (yield from real(e))
  life._run_program_routine = program
  life.body.run(life.run_errand_routine(errand))
  assert order == ["undock"] and seen == [False]

  # ...and with no plug in, nothing changes
  order.clear()
  life.body.run(life._after_decision_routine(ov.Decision(action="explore")))
  assert order == ["explore"]


def test_nothing_that_stands_still_takes_its_plug_out():
  """`idle`, `recall`, `look` and a claim move no part of it: the pack goes
  on filling (a mind may read a recall through before it chooses)."""
  life = _life()
  _plugged_in(life)
  life.body.undock_routine = lambda: pytest.fail("unplugged to stand still")
  life.body.run(life._after_decision_routine(ov.Decision(action="idle")))
  assert life.body.on_charger


# ---- what it is told ---------------------------------------------------------


def test_the_next_run_says_the_charge_was_cut_and_that_the_plug_is_in(tmp_path):
  """Two restarts read "it cut my charge short": one mid-walk, the plug
  nowhere near, and one like this. Which is the one thing the robot cannot
  tell from the line."""
  life = _stub(tmp_path)
  life.state = "CHARGE"
  snap = _saved(life, tmp_path)
  back = _stub(tmp_path)
  back.body.on_charger = True
  _prelude(back, snap)
  assert any("it cut my charge short, and my plug is still in" in ln for ln in _history(back))
  # ...and one that walked, its plug nowhere
  back = _stub(tmp_path)
  _prelude(back, snap)
  said = next(ln for ln in _history(back) if "it cut my charge short" in ln)
  assert "plug" not in said, said


def test_the_first_decision_of_a_run_knows_it_is_charging(tmp_path):
  """Nothing steps before the first decision of a run carried on, so
  `charging` -- which only the step set -- read false, and a robot plugged
  in was told it was not."""
  life = _stub(tmp_path)
  life.state = "CHARGE"
  snap = _saved(life, tmp_path)
  back = _stub(tmp_path)
  back.body.on_charger = True
  _prelude(back, snap)
  assert back.charging_now is True


# ---- what a robot plugged in is not -------------------------------------------


def test_a_robot_plugged_in_has_not_left_the_station_nor_lingers_at_it():
  """Both read `state` for "at the station": a robot thinking there with
  its plug in was counted as yielding it to the other robot, and logged as
  in the other's way, once a restart for every hour of charging."""
  a = SimpleNamespace(state="CHARGE", charging_now=True, low_battery_wh=0.95, root="pluggybot",
                      battery=SimpleNamespace(fraction=0.5, energy_wh=4.0))
  b = SimpleNamespace(state="EXPLORE", charging_now=False, low_battery_wh=0.95, root=SECOND.root,
                      battery=SimpleNamespace(fraction=0.1, energy_wh=0.4))
  watch = enc.Encounters.__new__(enc.Encounters)
  watch.lives, watch.on_event, watch.yields = (a, b), [], 0
  watch._at_bay = {"pluggybot": False, SECOND.root: False}
  watch._open = []
  events: list = []
  watch.on_event.append(events.append)
  data = SimpleNamespace(time=10.0)
  watch._sense_yields(data)
  a.state = "DECIDE"                               # a restart, its plug still in
  data.time = 11.0
  watch._sense_yields(data)
  assert events == [], "thinking plugged in is not leaving"
  a.charging_now = False                           # ...and unplugged, it has left
  data.time = 12.0
  watch._sense_yields(data)
  assert [e["phase"] for e in events] == ["yielded"]

  life = _life(peer=True)
  _at_the_station(life)
  life.state = "DECIDE"
  life.charging_now = True
  for k in range(int(lc.RACK_LINGER_S) + 3):
    life.data.time = float(k)
    life._rack_linger_step()
  assert not any("lingering" in ln for ln in life.log), life.log[-2:]
  # ...and unplugged at the same station it does linger
  life.charging_now = False
  for k in range(int(lc.RACK_LINGER_S) + 3):
    life.data.time = 1000.0 + float(k)
    life._rack_linger_step()
  assert any("lingering" in ln for ln in life.log)


# ---- what the run's end does for the rest of the loop -------------------------


def test_a_wait_for_a_taken_station_ends_with_the_run_and_strands_nobody():
  """The robot it waits for has its plug in and its day over, and will not
  leave in this run: left to its bound the wait held the run open 1386 s,
  and the loop with no mind then read it as a charge that failed, a death."""
  life = _life(peer=True)
  far = life.peers[0]
  sx, sy, _ = life.body.charge_standoff()
  far.pos[:] = [sx + 0.2, sy]
  life.max_sim_time = 30.0

  def still(seconds, *a, **kw):
    life.data.time += seconds
    return tick.result(None)
  life.body.hold_routine = still
  life.body.go_to_routine = lambda *a, **kw: tick.result(True)
  assert life.body.run(life.go_charge_routine()) is False
  assert float(life.data.time) == pytest.approx(30.0, abs=lc.BAY_POLL_S + 1.0)
  assert "run" in life.charge_failure and "1386" not in life.charge_failure, life.charge_failure

  # ...and the loop's own trip is no stuck death for it
  life = _life(peer=True)
  far = life.peers[0]
  sx, sy, _ = life.body.charge_standoff()
  far.pos[:] = [sx + 0.2, sy]
  life.max_sim_time = 30.0
  life.body.hold_routine = lambda seconds, *a, **kw: (
    setattr(life.data, "time", life.data.time + seconds), tick.result(None))[1]
  life.body.go_to_routine = lambda *a, **kw: tick.result(True)
  life.stranded = False
  life.body.run(life._charge_trip_routine())
  assert not life.stranded and life.dead is None


def test_a_job_runs_out_on_the_seam_once_its_holders_day_is_over(tmp_path):
  """Its holder's pass of the day loop is the only place a held job could
  run out, and a day over has none: on 11-07 three were held 1,481-3,871 s
  against their 960 s. The seam runs it out as on a charge trip (#602): no
  hand is on it."""
  life = _counter_life(tmp_path)
  life.continuing = True
  task = _offer(life, "open_drawer", minutes=1)
  assert life._claim_task(task.id)
  life.max_sim_time = float(life.data.time)
  day = life._day_routine((0.5, 3.0, 0.0), life.max_sim_time, 90.0)
  life.body.run(day)
  assert life.state == "DONE" and life.tasks.get(task.id).state == "active"
  life.data.time += 57.0
  life.body.run(life.body.hold_routine(2.0))
  assert life.tasks.get(task.id).state == "active", "its minute is not yet past"
  life.body.run(life.body.hold_routine(2.0))
  assert life.tasks.get(task.id).state == "failed"
  assert life.tasks.get(task.id).verdict["reason"].startswith("not graded within")
