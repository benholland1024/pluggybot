"""scripts/mechanism_spike.py (#469): what opens from lying, flown as a batch.

The rules pinned are the batch's own -- each flight in a process of its own
(`run_pool` says why) -- and the claw's, hanging still before it grips.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mechanism_spike.py"


@pytest.fixture(scope="module")
def spike():
  spec = importlib.util.spec_from_file_location("mechanism_spike", SCRIPT)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def test_every_flight_is_a_process_of_its_own_even_one_at_a_time(spike, monkeypatch):
  made = []

  class Pool:
    """Runs here, and keeps what it was built with."""

    def __init__(self, processes, **kwargs):
      made.append(kwargs)

    def __enter__(self):
      return self

    def __exit__(self, *exc):
      return False

    def map(self, fn, jobs, chunksize=1):
      return [fn(j) for j in jobs]

  monkeypatch.setattr(spike, "Pool", Pool)
  for n_jobs in (1, 6):
    assert spike.run_pool(abs, [-1, -2], n_jobs) == [1, 2]
  assert made == [{"maxtasksperchild": 1}] * 2


@pytest.mark.parametrize("kind", ["handle", "drawer", "lip"])
def test_the_claw_hangs_still_before_the_jaws_go_down_to_the_knob(spike, kind):
  # After the empty sweep's short approach at Kp 60 the claw swung 3 deg on
  # its peg; going straight down, its crossbar landed on the handle's
  # bracket and every Kp 60 fit set-out lost its grip (2026-10-06).
  moves = []
  hand = object.__new__(spike.Hand)
  hand.sc = SimpleNamespace(mech=SimpleNamespace(kind=kind), look=lambda: None,
                            path=lambda s: np.zeros(3), knob_in_claw=lambda: None,
                            gripped=lambda: True)
  hand.hand = SimpleNamespace(jaws_routine=lambda closed, **_: ("jaws", closed))
  hand.run = moves.append
  hand.to = lambda p, speed=0.08: moves.append(("to", round(float(p[2]), 3))) or True
  hand.hold = lambda s: moves.append(("hold", s))
  assert hand.take()
  # the approach, then the move that brings the jaws to the knob or the lip
  approach, onto = [k for k, m in enumerate(moves) if m[0] == "to"][:2]
  waits = [m[1] for m in moves[approach + 1:onto] if m[0] == "hold"]
  assert sum(waits) >= spike.SWING_SETTLE_S >= 1.0
