"""The record store (issue #221): unbounded, append-only, searchable,
and scoped to the living generation. No sim, no model."""

import pytest

from pluggybot.mind.memory import (
  ACTIVE, FIND_LIMIT, RETIRED, RecordStore, fts_query,
)

R = "pluggybot"


def test_a_retired_row_leaves_the_view_and_stays_findable():
  m = RecordStore()
  a = m.add(R, "core", "robot", "the far board is worth the drive", topic="Top_of_mind.md")
  b = m.add(R, "core", "robot", "carry pays little", topic="Top_of_mind.md")
  assert [r.id for r in m.active(R, "core")] == [a.id, b.id]
  m.retire(a.id, t=5.0)
  assert [r.id for r in m.active(R, "core")] == [b.id]
  gone = m.get(a.id)
  assert gone.status == RETIRED and gone.retired_t == 5.0 and gone.text == a.text
  # ...and `recall` is exactly for what was taken off the page.
  assert [r.id for r in m.find(R, "far board")] == [a.id]
  # Retiring twice, or what is not there, changes nothing and raises nothing.
  assert m.retire(a.id, t=9.0).retired_t == 5.0
  assert m.retire(999) is None


def test_a_true_death_starts_a_new_generation_and_keeps_the_rows():
  m = RecordStore()
  a = m.add(R, "history", "system", "woke up")
  m.add("r2_pluggybot", "history", "system", "the other robot woke up")
  assert m.generation(R) == 1
  assert m.archive(R, t=100.0) == 1
  assert m.generation(R) == 2
  assert m.active(R, "history") == [] and m.find(R, "woke") == []
  assert m.get(a.id) is not None                  # on the volume still
  assert m.count("r2_pluggybot", "history") == 1  # the other robot is untouched
  b = m.add(R, "history", "system", "woke up again")
  assert b.generation == 2 and [r.id for r in m.active(R, "history")] == [b.id]


def test_find_ranks_by_match_and_takes_its_filters():
  m = RecordStore()
  one = m.add(R, "history", "system", "chose draw on whiteboard_b", t=10)
  two = m.add(R, "history", "system", "draw on whiteboard_b failed: no route", t=20)
  m.add(R, "note", "robot", "durian smells", topic="fruit/durian", title="smell", t=30)
  said = m.add(R, "history", "visitor", "please draw a house on whiteboard_b", t=40)
  hits = m.find(R, "whiteboard_b failed route")
  assert hits[0].id == two.id                      # three words matched
  assert {r.id for r in hits} == {one.id, two.id, said.id}
  assert [r.id for r in m.find(R, "whiteboard_b", since=15)] == [said.id, two.id]
  assert [r.id for r in m.find(R, "whiteboard_b", writer="visitor")] == [said.id]
  assert [r.id for r in m.find(R, "draw", kind="note")] == []
  assert m.find(R, "") == [] and m.find(R, "nothing like this") == []
  assert len(m.find(R, "whiteboard_b", limit=1)) == 1 and FIND_LIMIT == 8


def test_a_query_is_quoted_so_the_models_words_cannot_be_syntax():
  # `AND` / `NEAR` / an unbalanced quote are FTS5 syntax; a model's sentence
  # is not. Every word is quoted and OR-joined.
  assert fts_query('why did draw AND "the pen fail') == '"why" OR "did" OR "draw" OR "AND" OR "the" OR "pen" OR "fail"'
  m = RecordStore()
  m.add(R, "history", "system", "the pen did not stow")
  assert len(m.find(R, 'pen NEAR("stow") AND')) == 1


def test_the_index_and_the_tail_read_off_the_living_rows():
  m = RecordStore()
  m.add(R, "note", "robot", "a", topic="tasks/draw", title="far board")
  m.add(R, "note", "robot", "b", topic="tasks/draw", title="near board")
  m.add(R, "note", "robot", "c", topic="visitors/ben", title="likes houses")
  m.add(R, "note", "robot", "kg", topic="findings/mass", title="block a", fields={"value": 0.12})
  index = m.topics(R, "note")
  assert list(index) == ["tasks/draw", "visitors/ben", "findings/mass"]
  assert [r.title for r in index["tasks/draw"]] == ["far board", "near board"]
  assert [r.title for r in m.active(R, "note", prefix="findings/")] == ["block a"]
  assert m.active(R, "note", prefix="findings/")[0].fields == {"value": 0.12}
  for i in range(5):
    m.add(R, "history", "system", f"line {i}")
  assert [r.text for r in m.tail(R, "history", 2)] == ["line 3", "line 4"]


def test_rows_survive_a_reopen_and_carry_their_cites(tmp_path):
  path = str(tmp_path / "memory.sqlite")
  m = RecordStore(path)
  h = m.add(R, "history", "system", "draw failed at the far board", t=1)
  p = m.add(R, "core", "robot", "the far board is not worth it", topic="Top_of_mind.md",
            cites=[h.id], t=2)
  m.close()
  again = RecordStore(path)
  back = again.get(p.id)
  assert back.cites == (h.id,) and back.status == ACTIVE and back.at
  assert [r.id for r in again.active(R, "core")] == [p.id]
  assert again.find(R, "far board")[0].id in (h.id, p.id)


def test_an_unknown_kind_is_refused_at_the_store():
  with pytest.raises(AssertionError):
    RecordStore().add(R, "opinion", "robot", "x")
