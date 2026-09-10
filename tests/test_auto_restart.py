"""A dead robot waits, visibly, and stands itself up (issue #143).

The deployed world runs continuously and, on the `autonomous` arm, its robot
dies most days -- A0 died on four days in five. Until this, a death meant a
robot lying on the floor until a person pressed reset, which is tolerable
briefly and annoying immediately.

⚠ THE LINE THIS FILE MOSTLY EXISTS FOR: **an auto-restart is not an
intervention.** A run with a non-empty `interventions` array is excluded from
survival statistics (docs/Evaluation.md §5), because an admin's hand
contaminates a survival number. World behaviour on a timer is not a hand --
and if it wrote an entry there, every deployed run and every multi-life run
would be silently disqualified, with the exclusion invisible because an entry
in that array is supposed to be believed.

⚠ AND IT IS NOT #136's TRUE DEATH. This one KEEPS the volume, so the next
life reads its predecessor's `History.md` death line on every decision --
which is the whole of what dying costs. True death archives it.
"""

import mujoco
import pytest

from pluggybot import lifecycle as lc
from pluggybot.lifecycle import AUTO_RESTART_BY, HubLifecycle, world_config
from pluggybot.mind import overseer as ov
from pluggybot.mind.inbox import Inbox
from pluggybot.mind.thoughts import HISTORY


def _life(world: str = "room_hub", restart_after_s=30.0, inbox=None,
          **kw) -> HubLifecycle:
  """A mortal lifecycle with a SHORT restart timer. 30 s rather than the
  shipped 300 so a test is seconds of sim rather than minutes of it; the
  number under test is the parameter, never the default."""
  cfg = world_config(world)
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  data = mujoco.MjData(model)
  life = HubLifecycle(model, data, realtime=False, world=world,
                      battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"], errand=False,
                      inbox=inbox, mortal=True,
                      restart_after_s=restart_after_s, **kw)
  life.mission.start_at(*cfg["start"])
  life.home_pose = tuple(cfg["start"])
  life.survival_since = float(data.time)
  return life


def _events(life) -> list[dict]:
  seen: list[dict] = []
  life.on_event.append(seen.append)
  return seen


def _kill(life) -> None:
  life.battery.energy_wh = 0.0
  life.mission._drive(0.5, 0.0, 0.0)
  assert life.dead is not None


# ---- it stands up --------------------------------------------------------


def test_a_dead_robot_stands_itself_up_after_the_delay():
  """The acceptance case: the origin, a full pack, and a clock the robot
  ran itself. ⚠ SIM seconds, never wall -- the deployed world is paced to
  real time so the two agree there, but an experiment run is not, and a
  countdown that took five WALL minutes at 3x real time would be a
  different world."""
  life = _life(restart_after_s=20.0)
  seen = _events(life)
  life.mission._drive(2.0, 0.0, 0.0)
  moved = tuple(life.data.qpos[:3])
  _kill(life)
  died_at = life.dead["t"]
  # Still down well after the death and well before the delay is up. (The
  # STATE is whatever the death interrupted -- `DEAD` is set by the run
  # loop's `_wait_dead`, and this drives the mission directly.)
  life.mission._drive(10.0, 0.0, 0.0)
  assert life.dead is not None
  # ...and up on the far side of it.
  life.mission._drive(12.0, 0.0, 0.0)
  assert life.dead is None
  reset = next(e for e in seen if e["type"] == "reset")
  assert reset["t"] - died_at == pytest.approx(20.0, abs=0.5)
  assert reset["wasDead"] == "flat"
  # Refilled to full and then driven for a moment: `> 0.9` for the reason
  # `test_a_dead_robot_with_an_inbox_waits_and_a_reset_resumes_the_day`
  # uses it -- the motors draw the instant the robot is back on its feet.
  assert life.battery.fraction > 0.9
  assert tuple(life.data.qpos[:3]) != moved, "it went back to the start pose"
  # The survival clock restarts, which is what makes each life a data point
  # rather than one long span with a gap in it.
  #
  # ⚠ It starts a second LATER than the `reset` event's `t`, and that is
  # #107's ordering rather than anything here: the event is stamped when the
  # reset is decided, and `start_at` ends with a one-second settle drive
  # before the clock is re-seeded. The robot is not counted as awake while
  # it is being put down.
  since_reset = life.data.time - reset["t"]
  assert 0.0 < life.survival_s <= since_reset
  assert life.survival_s < died_at, "a NEW life, not the old clock resumed"


def test_the_delay_is_a_parameter_and_none_is_the_old_behaviour():
  """OFF by default, everywhere but `serve.py`. Every world before this
  waited for a person, for ever if need be, and a test or a demo must go on
  doing that."""
  assert lc.RESTART_AFTER_S == 300.0, "five minutes, and it is a parameter"
  life = _life(restart_after_s=None)
  assert life.restart_after_s is None
  _kill(life)
  life.mission._drive(20.0, 0.0, 0.0)
  assert life.dead is not None, "with no timer it waits for a person"
  assert life.reset_in_s is None
  # ...and the default constructor argument is that, not the constant.
  cfg = world_config("room_hub")
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  bare = HubLifecycle(model, mujoco.MjData(model), realtime=False,
                      world="room_hub", rack=cfg["rack"], errand=False)
  assert bare.restart_after_s is None


# ---- the countdown -------------------------------------------------------


def test_the_remaining_time_is_on_the_wire_and_counts_down_to_zero():
  """The site draws a countdown over the body from this (companion issue
  rooftop-media-2026 #215), so it has to reach zero at the reset rather
  than blink out from some remainder."""
  life = _life(restart_after_s=20.0)
  assert life.telemetry_status()["survival"].get("resetInS") is None, \
      "nothing to count while it is alive"
  _kill(life)
  first = life.telemetry_status()["survival"]["resetInS"]
  assert first == pytest.approx(20.0, abs=0.6)
  life.mission._drive(8.0, 0.0, 0.0)
  later = life.telemetry_status()["survival"]["resetInS"]
  assert later < first and later == pytest.approx(12.0, abs=0.6)
  # ...and it is gone once the robot is up, rather than sitting at zero.
  life.mission._drive(14.0, 0.0, 0.0)
  assert life.dead is None
  assert "resetInS" not in life.telemetry_status()["survival"]


def test_the_countdown_is_absent_rather_than_null_with_no_timer():
  """⚠ ABSENT, not null. "This robot is alive" and "this world has no
  restart timer" are both simply no number, and a `null` would be a
  countdown every consumer had to special-case before rendering."""
  life = _life(restart_after_s=None)
  _kill(life)
  assert "resetInS" not in life.telemetry_status()["survival"]
  assert life.telemetry_status()["survival"]["dead"] == "flat"


def test_the_countdown_never_goes_negative():
  """A step lands wherever it lands, so the clock can be read past the
  deadline before the seam next runs -- and a negative countdown says the
  restart has not happened when it is one step away."""
  life = _life(restart_after_s=5.0)
  _kill(life)
  life.dead["t"] -= 60.0        # as if it died a minute ago
  assert life.reset_in_s == 0.0


# ---- it is not an intervention -------------------------------------------


def test_an_auto_restart_is_not_an_intervention_and_an_admin_reset_is():
  """⚠ THE LINE. `rollup.excluded_because` drops a run with a non-empty
  `interventions` array from survival statistics, so world behaviour that
  wrote one would silently disqualify every deployed run -- invisibly,
  because the entry is supposed to be believed.

  Both halves in one test, because the claim is a DIFFERENCE: the timer
  leaves the array empty and an admin moving a LIVING robot fills it."""
  life = _life(restart_after_s=10.0, inbox=Inbox())
  seen = _events(life)
  _kill(life)
  life.mission._drive(12.0, 0.0, 0.0)
  assert life.dead is None, "the timer fired"
  assert life.interventions == [], \
      "an auto-restart is world behaviour, never an admin's hand"
  assert [e["type"] for e in seen if e["type"] == "intervention"] == []
  reset = next(e for e in seen if e["type"] == "reset")
  assert reset["intervention"] is False and reset["auto"] is True
  assert reset["by"] == AUTO_RESTART_BY
  # ...and the admin's hand on a LIVING robot still fills it.
  life.inbox.offer({"type": "reset_robot", "id": "rr_9", "from": "ben"})
  life._visitor_step()
  assert [i["what"] for i in life.interventions] == ["reset_robot"]
  admin = [e for e in seen if e["type"] == "reset"][-1]
  assert admin["intervention"] is True and admin["auto"] is False


def test_an_admin_rescue_and_a_timer_rescue_differ_only_in_who():
  """A rescue was never an intervention (issue #107) -- what #143 adds is a
  second party who can perform one. Both leave a `reset`, neither leaves an
  `intervention`, and `auto` is what tells them apart without reading
  prose."""
  for auto in (False, True):
    life = _life(restart_after_s=10.0, inbox=Inbox())
    seen = _events(life)
    _kill(life)
    if auto:
      life.mission._drive(12.0, 0.0, 0.0)
    else:
      life.inbox.offer({"type": "reset_robot", "id": "rr_1", "from": "ben"})
      life._visitor_step()
    reset = next(e for e in seen if e["type"] == "reset")
    assert life.dead is None and life.interventions == []
    assert reset["intervention"] is False and reset["auto"] is auto
    assert reset["by"] == (AUTO_RESTART_BY if auto else "ben")


def test_the_world_cannot_stand_up_a_living_robot():
  """Structural rather than defensive: the timer only ever fires on a dead
  robot, so `stand_up(auto=True)` can never reach the intervention branch.
  The assert is what says so out loud, and this is what keeps it honest."""
  life = _life(restart_after_s=10.0)
  life.mission._drive(20.0, 0.0, 0.0)
  assert life.dead is None
  assert life.interventions == []
  with pytest.raises(AssertionError, match="only follow a death"):
    life.stand_up(AUTO_RESTART_BY, auto=True)


# ---- what the next life knows -------------------------------------------


def test_the_death_line_survives_the_restart_and_the_next_life_reads_it():
  """The whole cost of dying (Evaluation.md §6), and the reason this is not
  #136's TRUE death: the volume is KEPT, so the same robot has to live with
  it. `History.md` is append-only and the robot is shown it on every later
  decision."""
  life = _life(restart_after_s=10.0)
  _kill(life)
  life.mission._drive(12.0, 0.0, 0.0)
  assert life.dead is None
  history = life.thoughts.volatile()[HISTORY]
  assert any("died" in line and "flat" in line for line in history)
  assert any("stood back up on my own" in line for line in history)
  # ...and the mind sees both on its very next decision.
  ctx = ov.context_for(life, thoughts=life.thoughts)
  assert any("died" in line for line in ctx["thoughts"][HISTORY])
  assert ctx["survival"]["deaths"] == 1, "the death is not forgotten either"
  assert ctx["survival"]["aliveS"] < 5.0, "but the clock is a NEW life"


# ---- the refusals it inherits -------------------------------------------


def test_the_timer_waits_for_a_seated_module_rather_than_yanking_it(monkeypatch):
  """`reset_robot`'s rule, and the timer RETRIES rather than reporting a
  refusal: a robot that died with the pen on its fork must not have it
  pulled out of the coupling, but it must still get up once the errand has
  put the thing down.

  ⚠ The seam RECOMPUTES `tool_powered` from contacts on every step, so the
  seated module is faked where the real one is read (test_reset_robot.py's
  own trick) rather than by setting the attribute, which the next step
  would overwrite."""
  life = _life(restart_after_s=5.0)
  seated = [True]
  monkeypatch.setattr(lc, "module_power_contact", lambda *a, **k: seated[0])
  _kill(life)
  assert life.tool_powered, "the seam did not see the seated module"
  life.mission._drive(10.0, 0.0, 0.0)
  assert life.dead is not None, "refused while the module is seated"
  assert life.reset_in_s == 0.0, "the clock is up; it is the fork that waits"
  seated[0] = False
  life.mission._drive(1.0, 0.0, 0.0)
  assert life.dead is None, "and it retried rather than giving up"
