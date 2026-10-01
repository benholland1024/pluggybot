"""Guards for the pack's energy book (power.py's `Pack`) and the test
suite's charge-rate knob. What a body DRAWS is its own (the quadruped's:
`test_quadruped.py::test_the_pack_draws_what_the_drivers_do`); the book is
every body's, and pinned here with direct calls."""

import pytest

from pluggybot.power import Pack
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path

#: A body's steady draw and its charger's rate, W: the charger out-supplies
#: the draw, and the draw is a real share of it.
DRAW_W, CHARGER_W = 35.0, 55.0


def _pack(**kw) -> Pack:
  pack = Pack(draw_w=DRAW_W, **kw)
  pack.charge_w = CHARGER_W
  return pack


def test_the_pack_drains_drawing_and_charges_docked():
  pack = _pack(capacity_wh=1.0, fraction=0.5)
  start = pack.energy_wh
  for _ in range(500):
    pack.update(None, 0.002)
  assert pack.energy_wh == pytest.approx(start - DRAW_W * 1.0 / 3600.0)
  drained = pack.energy_wh
  for _ in range(500):
    pack.update(None, 0.002, charging=True)
  # Net inflow must be positive but BELOW the gross charge rate: the body
  # keeps drawing while it charges (a real charger pays that too).
  gained_w = (pack.energy_wh - drained) * 3600 / 1.0    # 1 s
  assert gained_w == pytest.approx(CHARGER_W - DRAW_W)


def test_the_pack_clamps_at_empty_and_full():
  pack = _pack(capacity_wh=0.001, fraction=1.0)
  for _ in range(3000):
    pack.update(None, 0.002)
  assert pack.energy_wh == 0.0 and pack.empty
  for _ in range(30000):
    pack.update(None, 0.002, charging=True)
  assert pack.energy_wh == pack.capacity_wh
  assert pack.fraction == 1.0


# ---- the test suite's charge-rate knob (issue #84) ---------------------------


def test_the_scale_multiplies_the_net_fill_rate_exactly():
  """`charge_scale=k` fills the pack k times faster. Exactly k, not roughly.

  ⚠ THE SCALE MULTIPLIES THE NET, and this test exists because the obvious
  implementation does not. Charging is the charger's rate less whatever the
  body draws meanwhile -- here 55 W less 35, 20 W net -- and scaling the
  charger's rate itself by 2 would give (110 - 35) = 75 W, nearly FOUR
  times the fill. A knob nobody can predict from the number they typed is
  worse than no knob, because the suite's clock is exactly what it is
  meant to be reasoned about.
  """
  rates = {}
  for k in (1.0, 2.0, 5.0):
    pack = _pack(capacity_wh=10.0, fraction=0.0, charge_scale=k)
    pack.update(None, 1.0, charging=True)
    rates[k] = pack.energy_wh
  assert rates[2.0] == pytest.approx(rates[1.0] * 2.0)
  assert rates[5.0] == pytest.approx(rates[1.0] * 5.0)


def test_scale_one_is_arithmetically_the_old_behaviour():
  """The whole reason every existing mission, recording and measurement can
  stand unchanged: at 1.0 the expression IS `p -= charge_w`."""
  scaled = _pack(capacity_wh=10.0, fraction=0.5, charge_scale=1.0)
  scaled.update(None, 0.5, charging=True)
  plain = _pack(capacity_wh=10.0, fraction=0.5)
  plain.update(None, 0.5, charging=True)
  assert scaled.energy_wh == plain.energy_wh
  assert scaled.last_power_w == plain.last_power_w


def test_a_scale_that_is_not_a_positive_number_is_refused():
  """Zero would be a pack that never fills and a charge that never ends;
  negative would be a charger that drains. Both are typos, and both would
  present as a mission that hangs at the dock."""
  for bad in (0.0, -1.0):
    with pytest.raises(ValueError, match="charge_scale"):
      Pack(capacity_wh=1.0, charge_scale=bad)


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
  from pluggybot.power import charge_scale_from_env

  assert charge_scale_from_env() == 1.0
  assert Pack().charge_scale == 1.0
  assert stub_life().battery.charge_scale == 1.0

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
  from pluggybot.lifecycle import CHARGE_TIMEOUT_MIN, QUAD_HOME, world_config

  served_wh = world_config(QUAD_HOME)["hosting_battery_wh"]

  def timeout_at(scale: float) -> float:
    return stub_life(battery_wh=served_wh, charge_scale=scale).charge_timeout

  # The served pack: big enough that the cap is the computed need, not the floor.
  assert timeout_at(4.0) > CHARGE_TIMEOUT_MIN
  assert timeout_at(4.0) == pytest.approx(timeout_at(1.0) / 4.0)
