"""scripts/mechanism_spike.py (#469): what opens from lying, flown as a batch.

What is pinned is what cost a night: a flight builds the house, a body and
its renderers, and a world dropped in a worker that lives on was held until
the cycle collector ran -- 300 flights through six reused workers took the
dev box into swap and its desktop down (2026-10-06).
"""

import importlib.util
from pathlib import Path

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
