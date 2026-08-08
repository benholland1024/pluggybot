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
