"""Guards for the battery model (power.py) and the CHARGE state (milestone 7)."""

import mujoco
import pytest

from pluggybot.power import (
  CHARGE_W, DEMO_CAPACITY_WH, ELECTRONICS_W, NOMINAL_V, STALL_A, Battery,
)


def drive(model, data, rads: float, steps: int) -> None:
  data.ctrl[model.actuator("left_motor").id] = rads
  data.ctrl[model.actuator("right_motor").id] = rads
  for _ in range(steps):
    mujoco.mj_step(model, data)


def test_idle_robot_draws_electronics_only(world_model, world_data):
  for _ in range(500):
    mujoco.mj_step(world_model, world_data)
  battery = Battery(world_model)
  battery.update(world_data, 0.002)
  assert battery.last_power_w == pytest.approx(ELECTRONICS_W, abs=1.5), \
    "a parked robot should draw only the electronics load"


def test_cruise_power_is_honest(world_model, world_data):
  """Cruising draw must land in the plausible range for two 12 V gearmotors
  plus a Pi -- not milliwatts (drain would never trigger) and nowhere near
  the both-motors-stalled worst case (~130 W)."""
  battery = Battery(world_model)
  drive(world_model, world_data, 8.0, 1500)     # ~0.36 m/s cruise, settled
  battery.update(world_data, 0.002)
  assert ELECTRONICS_W + 1.0 < battery.last_power_w < 60.0
  worst = ELECTRONICS_W + 2 * NOMINAL_V * (STALL_A + 0.2)
  assert battery.last_power_w < worst / 2


def test_battery_drains_driving_and_charges_docked(world_model, world_data):
  battery = Battery(world_model, capacity_wh=DEMO_CAPACITY_WH, fraction=0.5)
  start = battery.energy_wh
  for _ in range(500):
    mujoco.mj_step(world_model, world_data)
    battery.update(world_data, 0.002)
  assert battery.energy_wh < start, "driving must cost energy"
  drained = battery.energy_wh
  for _ in range(500):
    mujoco.mj_step(world_model, world_data)
    battery.update(world_data, 0.002, charging=True)
  assert battery.energy_wh > drained, "charging must add energy"
  # Net inflow must be positive but BELOW the gross charge rate: the
  # electronics keep drawing while plugged in (a real charger pays that too).
  gained_w = (battery.energy_wh - drained) * 3600 / 1.0    # 1 s of sim
  assert 0.0 < gained_w < CHARGE_W


def test_battery_clamps_at_empty_and_full(world_model, world_data):
  battery = Battery(world_model, capacity_wh=0.001, fraction=1.0)
  drive(world_model, world_data, 8.0, 1000)
  for _ in range(3000):
    battery.update(world_data, 0.002)
  assert battery.energy_wh == 0.0 and battery.empty
  for _ in range(30000):
    battery.update(world_data, 0.002, charging=True)
  assert battery.energy_wh == battery.capacity_wh
  assert battery.fraction == 1.0


# ---- the test suite's charge-rate knob (issue #84) ---------------------------


def test_the_scale_multiplies_the_net_fill_rate_exactly(world_model, world_data):
  """`charge_scale=k` fills the pack k times faster. Exactly k, not roughly.

  ⚠ THE SCALE MULTIPLIES THE NET, and this test exists because the obvious
  implementation does not. Charging is `CHARGE_W` (55 W) less whatever the
  held press draws (~35 W stalled), so ~19 W net -- and scaling `CHARGE_W`
  itself by 2 would give (110 - 35) = 75 W, nearly FOUR times the fill. A
  knob nobody can predict from the number they typed is worse than no knob,
  because the suite's clock is exactly what it is meant to be reasoned about.
  """
  rates = {}
  for k in (1.0, 2.0, 5.0):
    battery = Battery(world_model, capacity_wh=10.0, fraction=0.0,
                      charge_scale=k)
    battery.update(world_data, 1.0, charging=True)
    rates[k] = battery.energy_wh
  assert rates[2.0] == pytest.approx(rates[1.0] * 2.0)
  assert rates[5.0] == pytest.approx(rates[1.0] * 5.0)


def test_scale_one_is_arithmetically_the_old_behaviour(world_model, world_data):
  """The whole reason every existing mission, recording and measurement can
  stand unchanged: at 1.0 the new expression IS `p -= CHARGE_W`."""
  scaled = Battery(world_model, capacity_wh=10.0, fraction=0.5, charge_scale=1.0)
  scaled.update(world_data, 0.5, charging=True)
  plain = Battery(world_model, capacity_wh=10.0, fraction=0.5)
  plain.update(world_data, 0.5, charging=True)
  assert scaled.energy_wh == plain.energy_wh
  assert scaled.last_power_w == plain.last_power_w


def test_a_scale_that_is_not_a_positive_number_is_refused(world_model):
  """Zero would be a pack that never fills and a charge that never ends;
  negative would be a charger that drains. Both are typos, and both would
  present as a mission that hangs at the dock."""
  for bad in (0.0, -1.0):
    with pytest.raises(ValueError, match="charge_scale"):
      Battery(world_model, capacity_wh=1.0, charge_scale=bad)


def test_the_deployed_default_is_one():
  """⚠ THE ONE THAT MATTERS FOR HONESTY. The multiplier is for the suite and
  for nothing else: charging is a SCORED task and economy/metabolism.json was
  calibrated against measured throughput at the honest rate (102 points per
  sim-hour), so a faster charge means more cycles an hour, more points, and a
  metabolism tuned against a world that does not exist.

  Checked three ways, because "the default is 1.0" can rot in three places:
  the constructor's default, the environment reader with nothing set, and the
  deployment's own configuration.
  """
  import mujoco

  from pluggybot.power import charge_scale_from_env
  from pluggybot.lifecycle import HubLifecycle, world_config

  assert charge_scale_from_env() == 1.0

  cfg = world_config("home")
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  life = HubLifecycle(model, mujoco.MjData(model), realtime=False,
                      world="home", rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"])
  assert life.battery.charge_scale == 1.0

  # ...and nothing in the deployment sets it. The sim service's environment
  # is the list in compose.yaml; a scale there would be invisible from Python
  # and would silently re-price the world the site serves.
  from pathlib import Path
  repo = Path(__file__).parent.parent
  for name in ("Dockerfile", "deploy/entrypoint.sh"):
    text = (repo / name).read_text()
    assert "PLUGGY_CHARGE_SCALE" not in text, \
        f"{name} sets the charge scale; the served world must run at 1.0"


def test_an_unreadable_scale_falls_back_to_one_out_loud(monkeypatch, capsys):
  """Failing SAFE here means failing HONEST. A typo that silently sped the
  world up would surface much later as a metabolism that no longer matches
  its own measurement, which is the hardest kind of bug to attribute."""
  from pluggybot.power import CHARGE_SCALE_ENV, charge_scale_from_env

  for bad in ("nonsense", "0", "-2"):
    monkeypatch.setenv(CHARGE_SCALE_ENV, bad)
    assert charge_scale_from_env() == 1.0
    assert CHARGE_SCALE_ENV in capsys.readouterr().out
  monkeypatch.setenv(CHARGE_SCALE_ENV, "4.5")
  assert charge_scale_from_env() == 4.5


def test_the_charge_cap_scales_with_the_rate():
  """⚠ A TIMEOUT THAT CANNOT FIRE IS NOT A TIMEOUT (issue #84's first
  warning). `charge_timeout` is already sized against the pack and the
  measured rate; the scale is one more factor in the same expression, or a
  five-times-faster charge gets a cap sized for the honest one and the guard
  against pressing on dead pins quietly stops guarding.
  """
  import mujoco

  from pluggybot.lifecycle import HubLifecycle, world_config

  cfg = world_config("home")
  model = mujoco.MjModel.from_xml_path(cfg["model"])

  def timeout_at(scale: float) -> float:
    life = HubLifecycle(model, mujoco.MjData(model), realtime=False,
                        world="home", battery_wh=40.0, rack=cfg["rack"],
                        grid_bounds=cfg["grid_bounds"],
                        low_battery_wh=cfg["low_battery_wh"],
                        charge_scale=scale)
    return life.charge_timeout

  # A pack big enough that the cap is the computed need rather than the floor.
  assert timeout_at(4.0) == pytest.approx(timeout_at(1.0) / 4.0)
