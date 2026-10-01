"""A dead robot waits, visibly, and stands itself up (issue #143).

The deployed world runs continuously and, on the `autonomous` arm, its robot
dies most days. Until this, a death meant a robot lying on the floor until a
person pressed reset, which is tolerable briefly and annoying immediately.

⚠ THE LINE THIS FILE MOSTLY EXISTS FOR: **an auto-restart is not an
intervention.** An admin's hand contaminates a survival number
(docs/Evaluation.md §5); world behaviour on a timer is not a hand -- and if
it wrote an entry in `interventions`, every deployed day would be silently
disqualified, with the exclusion invisible because an entry in that array
is supposed to be believed.

⚠ AND IT IS NOT #136's TRUE DEATH. This one KEEPS the volume, so the next
life reads its predecessor's `History.md` death line on every decision --
which is the whole of what dying costs. True death archives it.

The loop's bookkeeping, pinned on the stub body (`tests/test_body.py`);
the tool a stand-up takes home, on the served quadruped.
"""

import math

import mujoco
import pytest

from pluggybot import lifecycle as lc
from pluggybot.lifecycle import AUTO_RESTART_BY, QUAD_HOME, HubLifecycle, world_config
from pluggybot.mind import overseer as ov
from pluggybot.mind.inbox import Inbox
from pluggybot.mind.thoughts import HISTORY
from pluggybot.robot import world_spec
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path


#: The restart timer every test here runs on, and how long to hold to be
#: sure it has fired. SHORT on purpose (issue #158): the claims are about
#: the ORDER of things -- still down before the delay, up after it, the
#: event stamped at the delay -- and none of them depends on the delay's
#: size; the number under test is the parameter, never the default.
TIMER_S = 3.0
PAST_S = TIMER_S + 1.0


def _born(life) -> HubLifecycle:
  start = world_config(QUAD_HOME)["start"]
  life.body.start_at(*start)
  life.home_pose = tuple(start)
  life.survival_since = float(life.data.time)
  return life


def _life(restart_after_s=TIMER_S, inbox=None, **kw) -> HubLifecycle:
  """A mortal lifecycle on the stub with the short restart timer above."""
  return _born(stub_life(inbox=inbox, mortal=True,
                         restart_after_s=restart_after_s, **kw))


def _events(life) -> list[dict]:
  seen: list[dict] = []
  life.on_event.append(seen.append)
  return seen


def _hold(life, seconds: float) -> None:
  life.body.run(life.body.hold_routine(seconds))


def _kill(life) -> None:
  life.battery.energy_wh = 0.0
  _hold(life, 0.5)
  assert life.dead is not None


# ---- it stands up --------------------------------------------------------


def test_a_dead_robot_stands_itself_up_after_the_delay():
  """The acceptance case: the origin, a full pack, and a clock the robot
  ran itself. ⚠ SIM seconds, never wall -- the deployed world is paced to
  real time so the two agree there, but an experiment run is not, and a
  countdown that took five WALL minutes at 3x real time would be a
  different world."""
  life = _life()
  seen = _events(life)
  _hold(life, 4.0)
  life.body.x, life.body.y = 2.0, 2.5                # it went somewhere
  _kill(life)
  died_at = life.dead["t"]
  # Still down after the death and before the delay is up. (The STATE is
  # whatever the death interrupted -- `DEAD` is set by the run loop's
  # `_wait_dead`, and this drives the mission directly.)
  _hold(life, TIMER_S / 3)
  assert life.dead is not None
  # ...and up on the far side of it.
  _hold(life, PAST_S)
  assert life.dead is None
  reset = next(e for e in seen if e["type"] == "reset")
  assert reset["t"] - died_at == pytest.approx(TIMER_S, abs=0.5)
  assert reset["wasDead"] == "flat"
  # Refilled to full and then held for a moment: `> 0.9`, as the body may
  # draw the instant it is back on its feet.
  assert life.battery.fraction > 0.9
  assert life.body.pose == pytest.approx(life.home_pose), "back at the start pose"
  # The survival clock restarts, which is what makes each life a data point
  # rather than one long span with a gap in it.
  since_reset = life.data.time - reset["t"]
  assert 0.0 < life.survival_s <= since_reset + 1e-3   # the event's t is rounded
  assert life.survival_s < died_at, "a NEW life, not the old clock resumed"


def test_the_delay_is_a_parameter_and_none_is_the_old_behaviour():
  """OFF by default, everywhere but `serve.py`. Every world before this
  waited for a person, for ever if need be, and a test or a demo must go on
  doing that."""
  assert lc.RESTART_AFTER_S == 300.0, "five minutes, and it is a parameter"
  life = _life(restart_after_s=None)
  assert life.restart_after_s is None
  _kill(life)
  _hold(life, PAST_S)
  assert life.dead is not None, "with no timer it waits for a person"
  assert life.reset_in_s is None
  # ...and the default constructor argument is that, not the constant.
  assert stub_life().restart_after_s is None


# ---- the countdown -------------------------------------------------------


def test_the_remaining_time_is_on_the_wire_and_counts_down_to_zero():
  """The site draws a countdown over the body from this (companion issue
  rooftop-media-2026 #215), so it has to reach zero at the reset rather
  than blink out from some remainder."""
  life = _life()
  assert life.telemetry_status()["survival"].get("resetInS") is None, \
      "nothing to count while it is alive"
  _kill(life)
  first = life.telemetry_status()["survival"]["resetInS"]
  assert first == pytest.approx(TIMER_S, abs=0.6)
  _hold(life, 1.0)
  later = life.telemetry_status()["survival"]["resetInS"]
  assert later < first and later == pytest.approx(TIMER_S - 1.0, abs=0.6)
  # ...and it is gone once the robot is up, rather than sitting at zero.
  _hold(life, PAST_S)
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
  life = _life()
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
  life = _life(inbox=Inbox())
  seen = _events(life)
  _kill(life)
  _hold(life, PAST_S)
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
    life = _life(inbox=Inbox())
    seen = _events(life)
    _kill(life)
    if auto:
      _hold(life, PAST_S)
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
  life = _life()
  _hold(life, PAST_S)
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
  life = _life()
  _kill(life)
  _hold(life, PAST_S)
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


def test_the_timer_does_not_wait_behind_a_seated_module_and_takes_it_home(
    monkeypatch):
  """⚠ REVERSED BY ISSUE #348. The timer used to wait for the errand to put a
  seated tool down -- "wait until the errand returns", which Ben rejected
  (2026-09-24): behind an errand that never returns, the robot never gets
  up. The day loop now closes the routine a stand-up lands in
  (`test_stand_up.py`), so the timer fires at its time and the tool comes
  home with the robot (#311's rescue). Shown to fail with the wait put back.

  ⚠ The seam RECOMPUTES `tool_powered` from the coupling on every step, so
  the seated module is faked where the fork is read, rather than by setting
  the attribute, which the next step would overwrite."""
  cfg = world_config(QUAD_HOME)
  spec = world_spec(cfg["model"])
  model = spec.compile()
  life = _born(HubLifecycle(model, mujoco.MjData(model), realtime=False,
                            world=QUAD_HOME, spec=spec,
                            battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                            grid_bounds=cfg["grid_bounds"],
                            low_battery_wh=cfg["low_battery_wh"],
                            mortal=True, restart_after_s=1.0))
  try:
    monkeypatch.setattr(life.body.mission, "tool_powered", lambda m: m is not None)
    monkeypatch.setattr(life.body.mission, "seated_on", lambda m: life.root)
    assert life.module, "a lifecycle names the module it carries"
    adr = int(life.model.jnt_qposadr[int(life.model.body(life.module).jntadr[0])])
    life.data.qpos[adr:adr + 3] = (1.0, 1.0, 0.4)          # off its bay
    mujoco.mj_forward(life.model, life.data)
    home = list(life.model.qpos0[adr:adr + 3])
    _kill(life)
    assert life.tool_powered, "the seam did not see the seated module"
    _hold(life, 1.5)
    assert life.dead is None, "the timer waited behind the seated module"
    assert math.dist(life.data.qpos[adr:adr + 3], home) < 0.05, \
        "the tool was left where the robot fell"
  finally:
    life.body.close()
