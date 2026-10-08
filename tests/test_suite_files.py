"""Where the suite writes its files (tests/conftest.py): in memory where the
machine has room, because an fsync on a spinning disk, paid wherever the
collector closes a test's leftover memory store, stalled other tests (#473)."""

import os
from pathlib import Path

import conftest


def test_the_suites_files_are_in_memory_where_there_is_room(tmp_path):
  assert conftest.memory_temp({"TMPDIR": "/elsewhere"}) is None, "a TMPDIR already set wins"
  assert conftest.memory_temp({}, shm=str(tmp_path / "absent")) is None
  assert conftest.memory_temp({}, shm=str(tmp_path), free_gb=1e9) is None, "no room"
  room = conftest.memory_temp({}, shm=str(tmp_path), free_gb=0)
  assert room is not None and Path(room).parent == tmp_path
  # the run: with room in memory it names a TMPDIR, and pytest's own files
  # are in it, not where the platform's default was first read
  temp = os.environ.get("TMPDIR")
  assert temp or conftest.memory_temp({}) is None, "room in memory, and the files left on disk"
  if temp:
    assert tmp_path.resolve().is_relative_to(Path(temp).resolve()), f"{tmp_path} is not in {temp}"
