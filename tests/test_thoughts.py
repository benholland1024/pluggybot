"""The thought files: named memory with per-file write permissions (#38),
views over one record store since #221.

Six documents, three answers to "who may write this", and the whole module
is the enforcement of that table plus the two places the enforcement could
be quietly wrong.

  THE PROMPT SPLIT. A writable file placed in the cached prefix does not cost
  cache hits -- `Overseer.system` is built once and reused verbatim, so a
  mid-run write cannot move it either way. It costs the memory WORKING: the
  model would see its files as they stood at mission start and never a word
  it wrote afterwards. Nothing raises, nothing looks wrong, and the only
  symptom is a robot that keeps re-learning the same thing.

  A REFUSAL NOBODY SEES. A robot whose memory silently stopped accepting
  writes looks exactly like a model with nothing to say.

Nothing here touches the network or steps physics except the three lifecycle
cases at the bottom, which are what prove the files are written by the thing
that actually runs the robot rather than by a test calling the API directly.
"""

import json
import re

import pytest

from pluggybot.mind import constitution
from pluggybot.mind import overseer as ov
from pluggybot.lifecycle import board_book, world_config
from pluggybot.mind.overseer import Menu, Overseer
from pluggybot.mind.thoughts import (
  FINDINGS, GOALS, HISTORY, HUMAN, MAIN, MAX_LINE_CHARS, NOTES, ROBOT,
  SYSTEM, NAMES, SPECS, TOP_OF_MIND, ThoughtFiles, ThoughtRefused,
)
from pluggybot.telemetry.protocol import (
  THOUGHT_VERBS,
  DEFAULT_ROBOT_NAME, ROBOT_ROOT, THOUGHT_FILES, THOUGHT_WRITERS,
)

from test_overseer import FakeClient, full


@pytest.fixture()
def files(tmp_path):
  return ThoughtFiles(tmp_path / "thoughts")


# ---- the permission table ----------------------------------------------------


def test_the_files_and_their_writers_are_the_wire_vocabulary():
  """One list, two repos. A rename here that missed protocol.py would put a
  document on the wire under a name no client has a renderer for."""
  assert NAMES == THOUGHT_FILES
  assert {s.writer for s in SPECS.values()} <= set(THOUGHT_WRITERS)
  # ⚠ ONE human file since issue #154: `Main.md` is the CONSTITUTION and
  # `Goals.md` became the robot's own. The ownership is the whole point of
  # that issue -- quality 5 of the mission is read off a file nobody but the
  # robot writes -- so it is pinned here rather than left to the table.
  assert SPECS[MAIN].writer == HUMAN
  assert SPECS[GOALS].writer == ROBOT
  assert SPECS[HISTORY].writer == SYSTEM
  assert SPECS[TOP_OF_MIND].writer == ROBOT
  assert SPECS[NOTES].writer == ROBOT and SPECS[FINDINGS].writer == ROBOT


@pytest.mark.parametrize("name,by", [
  (MAIN, ROBOT), (MAIN, SYSTEM),          # nothing writes the constitution
  (GOALS, HUMAN), (GOALS, SYSTEM),        # the goals are the ROBOT's (#154)
  (HISTORY, ROBOT),                       # the robot cannot edit its past
  (TOP_OF_MIND, SYSTEM),                  # ...and code does not think for it
  (NOTES, SYSTEM), (FINDINGS, HUMAN),
])
def test_a_write_by_the_wrong_writer_is_refused_and_visible(files, name, by):
  """The issue's first acceptance criterion. REFUSED AND VISIBLE are two
  claims: a silent no-op would leave a robot believing it had remembered
  something, and leave a human believing the file they edited is the file
  the robot reads."""
  before = files.read(name)
  with pytest.raises(ThoughtRefused) as e:
    files.append(name, "I hereby rewrite myself", by=by, t=1.0)
  assert name in str(e.value)
  assert files.read(name) == before, "a refused write changed the file"
  assert files.refusals and name in files.refusals[-1]
  assert files.writes[name] == 0


def test_the_robot_writes_only_its_own_documents(files):
  """...and every one of them is its own (issue #154)."""
  assert files.pin("whiteboard_b is the one people look at", t=2.0)
  assert files.read(TOP_OF_MIND) == "whiteboard_b is the one people look at"
  assert files.intend("ink both boards this week", t=3.0)
  assert files.read(GOALS) == "ink both boards this week"
  assert files.note({"topic": "boards", "title": "b", "text": "the far one"}, t=4.0)
  # The other two are unreachable from the robot's verbs: each names its
  # document in code, and the general `append` checks the table. There is
  # no verb that takes a document name from a model at all.
  assert files.writes[TOP_OF_MIND] == 1 and files.writes[GOALS] == 1
  assert files.writes[NOTES] == 1
  assert all(files.writes[n] == 0 for n in (MAIN, HISTORY))


def test_the_constitution_is_rendered_from_the_library_and_a_hand_edit_is_set_aside(tmp_path):
  """Since issue #263 the volume's `Main.md` is a VIEW of the library file
  the environment names, not the source: a hand edit is archived beside
  the fresh render and reported, never honoured -- the header names the
  constitution in force, and an override nobody can name would make that
  a lie. Before #263 this test asserted the opposite (the edit was read
  back), which is how Luca's constitution went stale."""
  root = tmp_path / "thoughts"
  files = ThoughtFiles(root)
  assert (root / MAIN).read_text() == files.constitution.text + "\n"
  assert files.constitution.name == "default" and files.constitution_change is None
  (root / MAIN).write_text("You are a careful robot who likes the garden.\n")

  reopened = ThoughtFiles(root)
  assert reopened.read(MAIN) == files.constitution.text
  assert (root / MAIN).read_text() == files.constitution.text + "\n"
  assert (root / "Main.1.md").read_text() == \
    "You are a careful robot who likes the garden.\n"
  change = reopened.take_constitution_change()
  assert change["why"] == "edited" and change["archived"] == "Main.1.md"
  assert change["from"] == {"name": None, "sha": constitution.sha_of(
    "You are a careful robot who likes the garden.")}
  assert change["to"] == files.constitution.as_dict()
  assert reopened.take_constitution_change() is None, "announced once"
  # A whole run's worth of the robot's own writing must not touch it.
  reopened.pin("the garden is bigger than it looks", t=3.0)
  reopened.intend("plant the empty row", t=3.5)
  reopened.remember("charged to 92%", t=4.0)
  assert (root / MAIN).read_text() == files.constitution.text + "\n"
  assert reopened.writes[MAIN] == 0


def test_an_old_volume_starts_blank_and_keeps_its_files_aside(tmp_path):
  """A pre-#221 volume carries the four files and no store. The robot's and
  the system's are put aside (the true-death path: `.1.md` beside a fresh
  one), nothing is imported -- so the observation period holds only what
  was written through the new verbs -- and the constitution, a text from
  before the library, is REPLACED by the library's and kept aside on the
  same terms (issue #263: this is Luca's stale `Main.md` on deploy)."""
  root = tmp_path / "thoughts"
  root.mkdir()
  (root / MAIN).write_text("You are a careful robot.\n")
  (root / GOALS).write_text("water the garden\n")
  (root / HISTORY).write_text("[t=1s] woke up\n")
  (root / "Knowledge_and_Opinions.md").write_text("bay C sticks\n")
  files = ThoughtFiles(root)
  assert files.read(MAIN) == files.constitution.text
  assert files.read(GOALS) == "" and files.read(HISTORY) == ""
  assert files.read(TOP_OF_MIND) == "" and files.records.count("pluggybot") == 0
  assert sorted(files.archived_at_start) == sorted(
    [GOALS, HISTORY, "Knowledge_and_Opinions.md"])
  for kept in ("Goals.1.md", "History.1.md", "Knowledge_and_Opinions.1.md", "Main.1.md"):
    assert (root / kept).exists(), f"{kept} was not kept on the volume"
  assert (root / "Main.1.md").read_text() == "You are a careful robot.\n"
  assert (root / MAIN).read_text() == files.constitution.text + "\n"
  assert files.constitution_change["why"] == "replaced"
  assert files.constitution_change["from"]["name"] is None
  # A second start finds the store and puts nothing else aside.
  files.intend("plant the empty row", t=1.0)
  again = ThoughtFiles(root)
  assert again.archived_at_start == [] and again.lines(GOALS) == ["plant the empty row"]


def test_a_fresh_deploy_finds_the_constitution_to_read(tmp_path):
  """The human half is only checkable if a person can FIND it, so `Main.md`
  is rendered to the volume from the constitution the robot is living by
  (issue #263: rendered, not copied -- an edit there is set aside).

  ⚠ AND `Goals.md` IS NOT ONE OF THEM SINCE ISSUE #154. The bootstrap exists
  so a PERSON finds a file to edit; the goals are the robot's, and laying out
  an empty file for them would invite exactly the edit the ownership split
  exists to stop. It appears when the robot writes its first goal, like the
  other two files nobody hand-edits.
  """
  root = tmp_path / "thoughts"
  files = ThoughtFiles(root)
  assert (root / MAIN).read_text().strip() == files.read(MAIN)
  assert json.loads((root / "Constitution.json").read_text())["name"] == "default"
  # The documents code and the robot write are NOT created up front: they
  # do not exist until there is something in them, and a bootstrap is not a
  # write. The store is.
  for name in NAMES:
    if name != MAIN:
      assert not (root / name).exists(), f"{name} was laid out for a human"
  assert (root / "memory.sqlite").exists()
  assert all(n == 0 for n in files.writes.values())
  files.intend("ink both boards this week", t=1.0)
  assert (root / GOALS).read_text().strip() == "ink both boards this week"


# ---- the caps ----------------------------------------------------------------


def test_history_rolls_and_top_of_mind_refuses(files):
  """Both are size caps and they fail in OPPOSITE directions, deliberately.

  History is a rolling record nothing curates, so the oldest line falls off
  the VIEW -- and since #221 only the view: every line stays in the store.
  Top of mind is the robot's own, and dropping the oldest line silently
  would mean a robot that believes it remembers something it does not; it
  is refused instead, loudly, and `unpin` is the robot's remedy.
  """
  for i in range(400):
    files.remember(f"something happened, number {i}", t=float(i))
  assert len(files.read(HISTORY)) <= SPECS[HISTORY].cap
  assert files.dropped[HISTORY] > 0
  assert "number 399" in files.read(HISTORY)
  assert "number 0" not in files.read(HISTORY), "an unbounded history"
  assert files.records.count("pluggybot", "history") == 400, "a line was lost"
  assert files.records.find("pluggybot", "number 0")[0].text.endswith("number 0")

  with pytest.raises(ThoughtRefused) as e:
    for i in range(400):
      files.pin(f"an opinion about board number {i}", t=float(i))
  assert "full" in str(e.value)
  assert len(files.read(TOP_OF_MIND)) <= SPECS[TOP_OF_MIND].cap
  assert files.dropped[TOP_OF_MIND] == 0, \
    "the robot's own file dropped a line it was never told about"


def test_a_long_line_is_trimmed_rather_than_refused(files):
  """A cap on ONE line is not the same failure as a full file: prose cut
  short is still prose, so it is trimmed at the write path -- the journal
  note's rule -- while a file with no room left is refused."""
  written = files.pin("x" * (MAX_LINE_CHARS * 3), t=1.0)
  assert len(written) == MAX_LINE_CHARS
  assert files.pin("", t=1.0) == "", "an empty line is cost with no content"
  assert files.writes[TOP_OF_MIND] == 1


def test_unpin_is_how_the_robot_changes_its_mind(files):
  files.pin("whiteboard_a is the one people look at", t=1.0)
  files.pin("bay C sticks a little", t=2.0)
  assert files.unpin("whiteboard_a is the one people look at", t=3.0)
  assert files.lines(TOP_OF_MIND) == ["bay C sticks a little"]
  # A quote that picks out exactly one line works; the point is that the
  # robot does not have to reproduce its own sentence to the character.
  assert files.unpin("bay C sticks", t=4.0)
  assert files.lines(TOP_OF_MIND) == []
  # ...and what it unpinned is RETIRED, not gone: `recall` can find it.
  gone = files.records.find("pluggybot", "bay C sticks")
  assert len(gone) == 1 and not gone[0].active


@pytest.mark.parametrize("quote,why", [
  ("a thing it never wrote", "nothing"),
  ("board", "2 lines"),
])
def test_an_unpin_that_does_not_pick_out_one_line_is_refused(files, quote, why):
  """Ambiguity and a miss are both refusals, because the alternative is a
  robot that asked to drop one belief and dropped another -- or believes it
  dropped one and did not."""
  files.pin("board a is nearly full", t=1.0)
  files.pin("board b is empty", t=2.0)
  with pytest.raises(ThoughtRefused) as e:
    files.unpin(quote, t=3.0)
  assert why in str(e.value)
  assert len(files.lines(TOP_OF_MIND)) == 2


def test_there_is_no_verb_that_rewrites_a_file(files):
  """"Append or patch, not blind rewrite" -- a full-rewrite verb lets one
  bad generation erase everything the robot knows. The public surface is
  two verbs and neither can empty the page in one call."""
  assert not hasattr(files, "write")
  assert not hasattr(files, "replace")
  verbs = {"append", "forget", "pin", "unpin", "remember", "record",
           "retract", "note", "unnote", "intend", "drop_goal"}
  assert verbs <= set(dir(files))
  # ...and the by-verb dispatcher (issue #217) reaches only those: every
  # verb the registry knows is an add or a remove on one document.
  from pluggybot.mind import text
  for verb, surface in text.BY_VERB.items():
    assert verb in (surface.add, surface.remove)


# ---- persistence -------------------------------------------------------------


def test_the_files_survive_a_restart(tmp_path):
  """World state, on the terms the boards and the ledger are: every mission
  end is a restart, so a memory that did not survive one is not memory."""
  root = tmp_path / "thoughts"
  first = ThoughtFiles(root)
  first.pin("the far whiteboard is not worth the trip", t=10.0)
  first.remember("drew a house on whiteboard_a", t=20.0)

  second = ThoughtFiles(root)
  assert second.read(TOP_OF_MIND) == "the far whiteboard is not worth the trip"
  assert "drew a house on whiteboard_a" in second.read(HISTORY)
  # ...and a second day appends to the first day's record rather than
  # starting a fresh one.
  second.remember("woke up in home", t=0.0)
  assert len(ThoughtFiles(root).lines(HISTORY)) == 2


def test_an_in_memory_set_writes_nothing(tmp_path):
  """Every unit test, spike and demo without a state directory. It must be
  usable and must not litter the working directory."""
  files = ThoughtFiles()
  files.pin("nothing to see", t=1.0)
  assert files.read(TOP_OF_MIND) == "nothing to see"
  assert files.records.path == ":memory:"
  assert not list(tmp_path.iterdir())


# ---- the wire ----------------------------------------------------------------


def test_a_document_goes_out_whole_and_says_who_writes_it(files):
  files.pin("the garden is bigger than it looks", t=5.0)
  msg = files.message(TOP_OF_MIND, t=5.0)
  assert msg["type"] == "thought" and msg["name"] == TOP_OF_MIND
  assert msg["writer"] == ROBOT and msg["cap"] == SPECS[TOP_OF_MIND].cap
  assert msg["text"] == files.read(TOP_OF_MIND)
  assert msg["robot"] == "pluggybot"
  # JSON-serialisable, like every other typed message.
  assert json.loads(json.dumps(msg)) == msg
  # All of them open a stream, in reading order, the rows after them (#238).
  assert [m.get("name") for m in files.messages(t=5.0)] == [*NAMES, None]


def test_every_change_is_published_as_it_happens(files):
  """The site cannot read these files -- it runs on a different box -- so a
  change that is not streamed is a change the Thoughts tab never shows."""
  seen = []
  files.on_event.append(seen.append)
  files.pin("bay C sticks", t=1.0)
  files.remember("charged to 92%", t=2.0)
  docs = lambda: [m["name"] for m in seen if m["type"] == "thought"]  # noqa: E731
  assert docs() == [TOP_OF_MIND, HISTORY]
  assert seen[1]["text"] == "bay C sticks"
  # A REFUSED write publishes nothing: the file did not change.
  with pytest.raises(ThoughtRefused):
    files.append(MAIN, "I hereby rewrite myself", by=ROBOT, t=3.0)
  assert len(docs()) == 2
  # ...and the robot's own goals stream exactly as its opinions do (#154).
  files.intend("ink both boards", t=4.0)
  assert docs() == [TOP_OF_MIND, HISTORY, GOALS]


# ---- the rows on the wire (issue #238) ----------------------------------------


def test_a_write_is_a_record_event_and_a_retire_is_a_second_one_for_the_same_id(files):
  """The site shows the memory AS IT IS STORED (rooftop-media-2026 #281):
  rows with the ids the robot cites, not files. So every write puts its
  ROW on the wire before the re-rendered view, and retiring it is a second
  `record` for the same id saying so -- never a deletion, because nothing
  is deleted."""
  seen = []
  files.on_event.append(seen.append)
  files.pin("bay C sticks", t=1.0, cites=["#7"])
  files.unpin("bay C sticks", t=4.0)
  assert [m["type"] for m in seen] == ["record", "thought", "record", "thought"]
  written, retired = seen[0]["record"], seen[2]["record"]
  assert written == {"id": written["id"], "t": 1.0, "kind": "core",
                     "writer": ROBOT, "topic": TOP_OF_MIND, "title": "",
                     "text": "bay C sticks", "fields": {}, "cites": [7],
                     "status": "active"}
  assert retired == {**written, "status": "retired", "retiredT": 4.0}
  # The MESSAGE's clock is when it happened, as on every message; the row
  # keeps its own `t` (when it was written) inside.
  assert (seen[0]["t"], seen[2]["t"]) == (1.0, 4.0)
  assert seen[2]["robot"] == "pluggybot" and json.loads(json.dumps(seen[2])) == seen[2]
  # Every kind rides: a note with its topic and title, a finding with its
  # parsed fields, a History line, and a think (beside its `journal`).
  files.note({"topic": "bays", "title": "c", "text": "sticks"}, t=5.0)
  files.record({"quantity": "mass", "value": 2, "unit": "kg"}, t=6.0)
  files.remember("woke up", t=7.0)
  files.think("hmm", t=8.0, why="idle")
  rows = [m["record"] for m in seen if m["type"] == "record"]
  assert [(r["kind"], r["topic"], r["title"]) for r in rows[2:]] == [
    ("note", "bays", "c"), ("note", "findings/general", "mass"),
    ("history", "", ""), ("think", "", "")]
  assert rows[3]["fields"] == {"quantity": "mass", "value": 2.0, "unit": "kg",
                               "method": ""}
  assert rows[4]["writer"] == SYSTEM and rows[5]["text"] == "hmm"
  assert [m["type"] for m in seen[-2:]] == ["record", "journal"]
  # A refused write puts no row on the wire: there is no row.
  n = len(seen)
  with pytest.raises(ThoughtRefused):
    files.unpin("nothing like this")
  assert len(seen) == n


def test_a_stream_opens_with_the_active_rows_and_history_cut_to_its_window():
  """The `records` snapshot is the `goals` slot's answer for the tables: a
  late joiner gets every active core and note row, History's newest
  `SNAPSHOT_HISTORY` and the thinks inside that window -- never a retired
  row (the `record` event for the retire already said so, and a joiner
  who missed it does not need it), never the whole store."""
  from pluggybot.mind.thoughts import SNAPSHOT_HISTORY
  files = ThoughtFiles()
  files.pin("kept", t=1.0)
  files.pin("gone", t=2.0)
  files.unpin("gone", t=3.0)
  files.intend("a goal", t=4.0)
  files.note({"topic": "bays", "title": "c", "text": "sticks"}, t=5.0)
  files.think("before the window", t=6.0)
  for i in range(SNAPSHOT_HISTORY + 5):
    files.remember(f"line {i}", t=10.0 + i)
    if i == 20:
      files.think("inside the window", t=30.5)
  msg = files.records_message(t=200.0)
  rows = msg["records"]
  assert msg["type"] == "records" and msg["robot"] == "pluggybot"
  assert msg["generation"] == 1 and msg["t"] == 200.0
  assert [r["text"] for r in rows if r["kind"] == "core"] == ["kept", "a goal"]
  assert all(r["status"] == "active" for r in rows)
  history = [r for r in rows if r["kind"] == "history"]
  assert len(history) == SNAPSHOT_HISTORY and history[0]["text"].endswith("line 5")
  assert [r["text"] for r in rows if r["kind"] == "think"] == ["inside the window"]
  assert [r["id"] for r in rows] == sorted(r["id"] for r in rows), "oldest first"
  # It closes what a stream opens with, after the six documents.
  opening = files.messages(t=200.0)
  assert [m["type"] for m in opening] == ["thought"] * len(NAMES) + ["records"]
  # A TRUE DEATH sends the next robot's tables: a new generation, no rows.
  seen = []
  files.on_event.append(seen.append)
  files.archive(t=300.0)
  fresh = [m for m in seen if m["type"] == "records"]
  assert len(fresh) == 1 and fresh[0]["generation"] == 2 and fresh[0]["records"] == []


# ---- the prompt cache, which is the issue's trap ------------------------------


def test_only_the_human_files_ride_the_cached_prefix():
  """The split, from both sides. `volatile` is derived from the same flag
  `stable` reads, so the halves cannot disagree -- and the way two
  hand-written lists WOULD disagree is a file that reaches the model
  through neither, which looks from outside exactly like a robot that never
  learns anything."""
  files = ThoughtFiles()
  # ⚠ ONE file rides the prefix since issue #154: `Goals.md` became the
  # ROBOT's, so it had to move to the volatile half with the other two the
  # robot and the code write. The flags are not independent -- a
  # robot-written file left in the prefix is shown as it stood at mission
  # start for the rest of the run.
  assert set(files.stable()) == {MAIN}
  assert set(files.volatile()) == {GOALS, HISTORY, TOP_OF_MIND, FINDINGS, NOTES}
  assert set(files.stable()) | set(files.volatile()) == set(NAMES)
  assert not set(files.stable()) & set(files.volatile())


def test_what_the_robot_writes_it_can_read_back_the_same_run():
  """⚠ THE ONE THAT ACTUALLY BREAKS, and it is not the one the issue
  predicts. A writable file placed in the stable half does NOT cost cache
  hits -- `Overseer.system` is built once and sent verbatim, so a mid-run
  write cannot move it whatever the split says. What it costs is the
  memory working at all: the model would be shown the file as it stood at
  mission start and never see a word it wrote afterwards, re-learning the
  same thing every hour. That is why the placement is guarded here rather
  than only by the byte-identical prefix test.
  """
  files = ThoughtFiles()
  files.pin("bay C sticks a little", t=1.0)
  files.remember("charged to 92%", t=2.0)
  files.intend("draw on the far board this week", t=3.0)
  files.note({"topic": "bays", "title": "bay C", "text": "it sticks"}, t=4.0)
  turn = ov._user_turn({"thoughts": files.volatile()})
  assert "bay C sticks a little" in turn
  assert "charged to 92%" in turn
  # ...and a note's INDEX, but not its body: that is a `recall` away.
  assert '"bays"' in turn and '"bay C"' in turn and "it sticks" not in turn
  # ...and the robot's GOALS, which is the same failure with higher stakes
  # (issue #154): a goal the model cannot read back is a goal it sets again
  # every hour, and quality 5 of the mission is measured off this file.
  assert "draw on the far board this week" in turn


def test_the_prefix_does_not_move_when_the_robot_writes():
  """The issue's second acceptance criterion, and the reason the split is
  by WRITER rather than by content: the two files that change during a run
  are exactly the two the robot and the code can write."""
  files = ThoughtFiles()
  menu = Menu(boards=("whiteboard_a",), programs=("house",))
  boss = Overseer(menu, thoughts=files, client=FakeClient())
  before = boss.system[0]["text"]

  files.pin("whiteboard_b is the one people look at", t=1.0)
  files.remember("drew a house on whiteboard_a", t=2.0)
  files.intend("keep both boards inked", t=3.0)

  assert boss.system[0]["text"] == before, \
    "a self-edit moved the cached prefix -- every call now pays full price"
  assert boss.system[0]["cache_control"] == {"type": "ephemeral"}
  # ...and the writable files are genuinely reaching the model, in the turn
  # AFTER the breakpoint. Absent from both halves would pass the assertion
  # above for the worst possible reason.
  #
  # ⚠ Their CONTENT, not their names: the rules block in the prefix talks
  # ABOUT `Top_of_mind.md`, which is stable text and exactly
  # right. The same distinction `test_the_stable_prefix_is_byte_identical`
  # draws between quoted JSON keys and prose.
  assert "whiteboard_b is the one people look at" not in before
  assert "drew a house on whiteboard_a" not in before
  assert "keep both boards inked" not in before
  turn = ov._user_turn({"thoughts": files.volatile()})
  assert "whiteboard_b is the one people look at" in turn
  assert "drew a house on whiteboard_a" in turn
  assert "keep both boards inked" in turn


def test_the_history_the_model_sees_is_the_tail(files):
  """The whole file is on disk and on the wire; the PROMPT gets the last few
  lines. The last dozen things that happened are context, and the hundred
  before them are input tokens on every call for the rest of the mission."""
  from pluggybot.mind.thoughts import HISTORY_SHOWN
  for i in range(HISTORY_SHOWN * 3):
    files.remember(f"thing number {i}", t=float(i))
  shown = files.volatile()[HISTORY]
  assert len(shown) == HISTORY_SHOWN
  assert "thing number 0" not in " ".join(shown)
  assert f"thing number {HISTORY_SHOWN * 3 - 1}" in " ".join(shown)
  # ...each line carrying its record id, which is what a `cites` names.
  assert all(re.match(r"^#\d+ \[t=", line) for line in shown)
  # ...and nothing was lost from the file itself to achieve that.
  assert len(files.lines(HISTORY)) > HISTORY_SHOWN


def test_another_constitution_is_the_one_thing_that_should_move_it(tmp_path):
  """The other side of the same coin: naming a different CONSTITUTION
  between runs SHOULD invalidate the cache, because the prefix genuinely
  changed. Not a bug being tolerated -- it is the reason the split is safe.

  ⚠ `Main.md` rather than `Goals.md` since issue #154: the goals are the
  robot's and no longer in the prefix at all, so an edit to them cannot move
  it. The document a human chooses is the one that can (issue #263: chosen
  from the library, no longer edited on the volume).
  """
  root = tmp_path / "thoughts"
  menu = Menu(boards=("whiteboard_a",), programs=("house",))
  first = Overseer(menu, thoughts=ThoughtFiles(root), client=FakeClient())
  swapped = ThoughtFiles(root, constitution=constitution.load("curious"))
  second = Overseer(menu, thoughts=swapped, client=FakeClient())
  assert first.system != second.system
  assert "Understand the world you are in" in second.system[0]["text"]
  assert swapped.constitution_change["why"] == "swapped"


def test_the_persona_is_the_library_file_rather_than_the_code(tmp_path):
  """The library is what makes 'who the robot is' choosable without a code
  change, which is the whole reason it is data (issue #263). If the prompt
  kept a hardcoded persona alongside it, naming a file would change
  nothing visible."""
  library = tmp_path / "constitutions"
  library.mkdir()
  (library / "serious.md").write_text("You are a very serious robot. Never dawdle.\n")
  charter = constitution.load("serious", library=library)
  boss = Overseer(Menu(boards=("a",), programs=("house",)),
                  thoughts=ThoughtFiles(tmp_path / "thoughts", constitution=charter),
                  client=FakeClient())
  text = boss.system[0]["text"]
  assert "You are a very serious robot." in text
  assert "two-wheeled robot" not in text, \
    "the default persona is still in the prompt beside the named one"


# ---- the robot's own name (issue #39) ----------------------------------------


def test_the_robot_is_told_its_name_and_not_its_species(monkeypatch):
  """⚠ THE ONE THE PERSONA GOT WRONG. #39 made `pluggybot` the SPECIES and
  the name a per-instance thing (`robot_display_name`), but `Main.md`'s
  default said "You are PluggyBot" and nothing ever handed the overseer the
  real name -- so a robot renamed to Luca introduced itself by its species,
  one panel below a website header reading "Luca the pluggybot", and could
  not recognise its own name if a visitor used it.
  """
  monkeypatch.setenv("PLUGGY_ROBOT_NAME", "Luca")
  boss = Overseer(Menu(boards=("a",), programs=("house",)), client=FakeClient())
  text = boss.system[0]["text"]
  assert boss.robot_name == "Luca"
  assert "Your name is Luca." in text
  # The species is still stated -- it is true, and the robot should know what
  # it is -- but never AS the name.
  assert ROBOT_ROOT in text
  assert "You are PluggyBot" not in text and "You are Pluggybot" not in text


def test_the_name_is_not_frozen_into_the_file_on_disk(tmp_path, monkeypatch):
  """The reason the name is stated in the prompt rather than written into
  `Main.md`: that file becomes a HUMAN's the moment it exists, so a name
  baked into its default would survive every later rename while
  `$PLUGGY_ROBOT_NAME` went on meaning something else."""
  root = tmp_path / "thoughts"
  monkeypatch.setenv("PLUGGY_ROBOT_NAME", "Luca")
  files = ThoughtFiles(root)
  assert "Luca" not in (root / MAIN).read_text(), "a name was frozen into the file"
  assert "Luca" not in files.read(MAIN)

  # ...and a rename between runs reaches the robot with no file to edit.
  first = Overseer(Menu(boards=("a",), programs=("house",)),
                   thoughts=ThoughtFiles(root), client=FakeClient())
  monkeypatch.setenv("PLUGGY_ROBOT_NAME", "Mika")
  second = Overseer(Menu(boards=("a",), programs=("house",)),
                    thoughts=ThoughtFiles(root), client=FakeClient())
  assert "Your name is Luca." in first.system[0]["text"]
  assert "Your name is Mika." in second.system[0]["text"]
  # A rename SHOULD move the cached prefix -- it is a different robot
  # talking, on the same terms as a human editing Goals.md.
  assert first.system != second.system


def test_an_unnamed_robot_still_gets_a_readable_name(monkeypatch):
  """The 0.10.0 degrade rule, one layer in: absent must resolve to the
  default, never to "Your name is ." -- the wire header does the same."""
  monkeypatch.delenv("PLUGGY_ROBOT_NAME", raising=False)
  boss = Overseer(Menu(boards=("a",), programs=("house",)), client=FakeClient())
  assert boss.robot_name == DEFAULT_ROBOT_NAME
  assert f"Your name is {DEFAULT_ROBOT_NAME}." in boss.system[0]["text"]


def test_the_name_reaches_a_served_robot(monkeypatch, tmp_path):
  """`overseer.build` is what a served world calls, and the name has to
  survive that hop -- it reached the RECORDER back in #39 and stopped
  there, which is why nothing caught this."""
  monkeypatch.delenv("PLUGGY_ROBOT_NAME", raising=False)
  boss = ov.build("room_hub", None, enabled=True, client=FakeClient(),
                  thoughts=ThoughtFiles(tmp_path / "thoughts"),
                  robot_name="Luca")
  assert boss.robot_name == "Luca"
  assert "Your name is Luca." in boss.system[0]["text"]


# ---- the robot's own writes, through a real decision --------------------------


def test_a_decision_can_pin_and_note_without_spending_a_turn():
  """`pin`/`unpin`/`note`/`unnote` are orthogonal to the action, like
  `respond_to`: a robot that had to spend its turn to write a line down
  writes fewer of them than it should."""
  menu = Menu(boards=("whiteboard_a",), programs=("house",))
  boss = Overseer(menu, client=FakeClient(
    full(action="draw", board="whiteboard_a", program="house",
         pin="people look at whiteboard_a more than b",
         note={"topic": "boards", "title": "a", "text": "the popular one"},
         cites="#3 #4")))
  decision = boss.decide({})
  assert decision.action == "draw"          # the action still happened
  assert decision.pin == "people look at whiteboard_a more than b"
  assert decision.note == {"topic": "boards", "title": "a", "text": "the popular one"}
  assert decision.cites == "#3 #4"
  assert decision.source == "llm"


def test_the_schema_offers_both_verbs_and_no_third(menu_home):
  """A model can add a line and remove a line. There is deliberately no
  parameter that names a FILE, so `Main.md` is not reachable from a decision
  at all -- the permission check is a backstop, not the only lock."""
  props = menu_home.schema()["properties"]
  assert "pin" in props and "unpin" in props
  assert "note" in props and "unnote" in props
  assert not [k for k in props if "file" in k.lower()]
  assert "pin" in menu_home.schema()["required"]
  # ...and `think` is the FIRST property (issue #221): constrained decoding
  # follows this order, so the reasoning precedes the action.
  assert list(props)[0] == "think"


@pytest.fixture(scope="module")
def menu_home():
  return Menu.for_world("home", board_book("home"))


def test_a_decision_writes_history_and_knowledge_through_the_mission(tmp_path):
  """The files are written by the thing that actually runs the robot, not by
  a test calling the API directly -- and both halves land: what the code
  recorded about the decision, and what the robot chose to keep from it."""
  from test_overseer import _lifecycle

  files = ThoughtFiles(tmp_path / "thoughts")
  boss = Overseer(Menu.for_world("room_hub", None), thoughts=files,
                  client=FakeClient(full(action="carry", reason="tidying up",
                                         pin="bay C sticks a little")))
  life = _lifecycle("room_hub", overseer=boss, thoughts=files, errand=False)
  said: list[str] = []
  life.say_hooks.append(lambda t, line: said.append(line))
  life.mission.start_at(*world_config("room_hub")["start"])
  try:
    life._decide()
  finally:
    life.mission.close()

  assert "chose carry: tidying up" in files.read(HISTORY)
  assert files.read(TOP_OF_MIND) == "bay C sticks a little"
  assert any(line.startswith("THOUGHT pin: bay C sticks") for line in said)
  # ...and it is on disk, because the next mission is a different process.
  assert "bay C sticks a little" in (tmp_path / "thoughts"
                                     / TOP_OF_MIND).read_text()
  assert ThoughtFiles(tmp_path / "thoughts").read(TOP_OF_MIND) == "bay C sticks a little"


def test_the_mission_cannot_write_the_files_it_does_not_own(tmp_path):
  """`_remember` is code writing History. The same lifecycle must not be
  able to reach Main.md or Goals.md, whatever it passes -- the permission
  table is enforced at the write path, not promised by its callers."""
  from test_overseer import _lifecycle

  files = ThoughtFiles(tmp_path / "thoughts")
  life = _lifecycle("room_hub", thoughts=files, errand=False)
  said: list[str] = []
  life.say_hooks.append(lambda t, line: said.append(line))
  life._remember("something happened")
  assert "something happened" in files.read(HISTORY)

  goals_before = files.read(GOALS)
  with pytest.raises(ThoughtRefused):
    files.append(GOALS, "the mission says otherwise", by=SYSTEM, t=1.0)
  assert files.read(GOALS) == goals_before


def test_a_refused_thought_is_narrated_rather_than_swallowed(tmp_path):
  """The issue's "fail loudly" half, at the seam that matters. A refusal
  nobody can see leaves a robot believing it remembered something, and the
  only symptom is a mind that never learns anything -- indistinguishable
  from a model with nothing to say."""
  from test_overseer import _lifecycle

  files = ThoughtFiles(tmp_path / "thoughts")
  life = _lifecycle("room_hub", thoughts=files, errand=False)
  said: list[str] = []
  life.say_hooks.append(lambda t, line: said.append(line))
  # Fill the robot's file, then ask it to learn one more thing.
  with pytest.raises(ThoughtRefused):
    for i in range(400):
      files.pin(f"an opinion about board number {i}", t=float(i))

  life._reconsider(ov.Decision(action="idle", pin="one thing too many"))
  assert any("THOUGHT refused" in line and "full" in line for line in said)
  assert "one thing too many" not in files.read(TOP_OF_MIND)
  # ...and a mission is never ended by a memory write.
  assert life.thoughts.refusals


def test_every_memory_write_is_narrated_in_the_one_shape_the_site_parses(tmp_path):
  """`THOUGHT <verb>: <line>` is a two-repo contract (issue #159): the
  website's observatory reads it into a `thought` row, because the documents
  ride the wire whole and WHEN a line was written is carried by this line
  alone. Pinned cheaply -- one lifecycle, four verbs and a refusal through
  `_reconsider` -- and the vocabulary is asserted EQUAL to `THOUGHT_VERBS`,
  so a fifth verb added to `_reconsider` without the constant fails here
  rather than silently vanishing from the site's history."""
  from test_overseer import _lifecycle

  files = ThoughtFiles(tmp_path / "thoughts")
  life = _lifecycle("room_hub", thoughts=files, errand=False)
  said: list[str] = []
  life.say_hooks.append(lambda t, line: said.append(line))

  life._reconsider(ov.Decision(action="idle", pin="bay C sticks",
                               intend="tidy bay C",
                               note={"topic": "bays", "title": "bay C",
                                     "text": "it sticks on the way in"},
                               record={"quantity": "block mass", "value": 0.42,
                                       "unit": "kg", "method": "the lift"}))
  life._reconsider(ov.Decision(action="idle", unpin="bay C sticks",
                               drop_goal="tidy bay C", retract="block mass",
                               unnote="bays/bay C"))
  life._reconsider(ov.Decision(action="idle", unpin="nothing says this"))

  lines = [line for line in said if line.startswith("THOUGHT")]
  shape = re.compile(r"^THOUGHT ([a-z_]+): (.+)$")
  seen = {}
  for line in lines:
    m = shape.match(line)
    assert m, f"not the shape the site parses: {line!r}"
    seen[m.group(1)] = m.group(2)
  assert set(seen) == set(THOUGHT_VERBS)
  assert seen["pin"] == "bay C sticks"
  assert seen["unpin"] == "bay C sticks"
  assert seen["note"] == "bays/bay C: it sticks on the way in" == seen["unnote"]
  assert seen["record"] == "block mass = 0.42 kg -- the lift" == seen["retract"]
  assert "nothing" in seen["refused"]


# ---- the robot's own goals (issue #154) --------------------------------------


def test_a_full_goals_file_refuses_out_loud_rather_than_dropping_one(files):
  """The issue's constraint, and the same one `Top_of_mind.md`
  has: silently dropping the oldest goal leaves the robot believing it still
  holds a goal it no longer has, which is worse than being told no.

  ⚠ THE OPPOSITE OF `History.md`, deliberately. That file ROLLS because it is
  a record nobody acts on; these two REFUSE because the robot is meant to
  curate them, and `drop_goal` is the remedy the prompt names.
  """
  cap = SPECS[GOALS].cap
  # Distinct lines, because `drop_goal` quotes ONE line and refuses an
  # ambiguous quote -- which is itself the behaviour the remedy depends on.
  i = 0
  while len(files.read(GOALS)) + 220 < cap:
    files.intend(f"goal number {i} " + "x" * 200, t=1.0)
    i += 1
  before = files.read(GOALS)
  with pytest.raises(ThoughtRefused) as e:
    files.intend(f"goal number {i} " + "x" * 200, t=2.0)
  assert "full" in str(e.value)
  assert files.read(GOALS) == before, "a refused goal changed the file"
  assert files.dropped[GOALS] == 0, "a goal was silently dropped"
  # ...and the remedy works: make room and the write lands.
  assert files.drop_goal("goal number 0 ", t=3.0)
  assert files.intend("one I actually mean", t=4.0)


def test_no_verb_replaces_the_goals_file(files):
  """One bad generation must not be able to wipe out what the robot has
  decided to do -- the argument that shaped `pin`/`unpin`, applied to the
  file that now carries the mission's fifth quality.

  Enforced by ABSENCE, so this is a grep over the surface a decision can
  reach rather than a call: there is no verb to call.
  """
  import inspect
  from pluggybot.mind import thoughts as th
  verbs = [n for n, _ in inspect.getmembers(th.ThoughtFiles, inspect.isfunction)
           if not n.startswith("_")]
  assert "intend" in verbs and "drop_goal" in verbs
  for banned in ("set_goals", "replace_goals", "rewrite", "clear_goals"):
    assert banned not in verbs
  # The write path itself only ever appends one line or removes one line.
  assert not hasattr(files, "write_goals")


def test_a_decision_can_set_and_drop_a_goal_without_spending_a_turn(files):
  """`intend` / `drop_goal` / `serves` are FIELDS, like `pin` and
  `unpin`: the robot sets a goal on the decision it was already making, so
  writing one down costs it nothing."""
  d = ov.Decision(action="draw", board="whiteboard_a",
                  intend="ink both boards this week",
                  serves="ink both boards this week")
  assert d.action == "draw", "the action was spent on paperwork"
  assert d.as_dict()["intend"] == "ink both boards this week"
  assert d.as_dict()["serves"] == "ink both boards this week"
  assert d.as_dict()["dropGoal"] == ""


def test_nothing_in_scoring_can_read_the_goals_file():
  """The mission's rule: self-conceived goals are NOT paid. A goal that
  earned points would be a reward table the robot writes itself, which is
  the one thing economy/scoring.py exists to prevent."""
  import ast
  from pathlib import Path
  root = Path(__file__).parent.parent / "src" / "pluggybot" / "economy"
  for path in root.glob("*.py"):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
      # An IMPORT of the memory module is the only way to reach the file,
      # and checking the syntax tree rather than the text is what keeps a
      # docstring that MENTIONS the goals from failing this.
      if isinstance(node, ast.ImportFrom) and node.module:
        assert "thoughts" not in node.module, \
          f"{path.name} imports the robot's memory"
      names = ([a.name for a in node.names]
               if isinstance(node, ast.Import) else [])
      assert not any("thoughts" in n for n in names), \
        f"{path.name} imports the robot's memory"
      if isinstance(node, ast.Attribute):
        assert node.attr not in ("intend", "drop_goal", "read_goals"), \
          f"{path.name} calls a goals verb"


def test_a_goal_written_in_one_mission_is_read_back_in_the_next(tmp_path):
  """⚠ THE RESTART IS THE WHOLE POINT (issue #154). The deployed world ends
  a mission and starts another every hour, and `/var/lib/pluggybot` is a
  VOLUME -- so a goals file that did not survive that boundary would give
  the robot an hour's memory of what it meant to do and no more, which is
  indistinguishable from not having the file.

  This is the ledger's and the boards' rule applied to the robot's goals:
  world state persists, run state does not. `History.md` carries the death
  line across for the same reason (Evaluation.md section 6).
  """
  root = tmp_path / "thoughts"
  first = ThoughtFiles(root)
  first.intend("get the far whiteboard inked before the week is out", t=10.0)
  first.intend("find out why bay C sticks", t=20.0)
  first.pin("whiteboard_b is worth the trip after all", t=30.0)

  # The mission ends. Nothing is handed over in memory -- the next one opens
  # the same directory from scratch, which is what a restart actually does.
  second = ThoughtFiles(root)
  assert second.lines(GOALS) == [
    "get the far whiteboard inked before the week is out",
    "find out why bay C sticks",
  ], "the robot woke up with no idea what it had decided to do"
  assert second.read(TOP_OF_MIND) == "whiteboard_b is worth the trip after all"

  # ...and it can still act on them: drop one it finished, add another, and
  # a third mission sees exactly that.
  second.drop_goal("find out why bay C sticks", t=40.0)
  second.intend("keep the garden surveyed", t=50.0)
  third = ThoughtFiles(root)
  assert third.lines(GOALS) == [
    "get the far whiteboard inked before the week is out",
    "keep the garden surveyed",
  ]
  # The goals reach the next decision's prompt, not just the next process.
  assert "keep the garden surveyed" in ov._user_turn({"thoughts": third.volatile()})


# ---- the notes tier (issue #221) ---------------------------------------------


def test_a_note_is_a_titled_line_in_a_topic_the_robot_names(files):
  """The index (topics and titles) is what the model sees every turn; the
  body is a `recall` away. The topic is the robot's to name, made safe to
  key by; a title already in the topic is refused (there is no replace)."""
  assert files.note({"topic": "Visitors / Ben", "title": "likes houses",
                     "text": "asked for a house on the far board twice"}, t=1.0) \
    == "visitors/ben/likes houses: asked for a house on the far board twice"
  files.note({"topic": "tasks/draw", "title": "far board", "text": "fails from the explore's end"}, t=2.0)
  assert files.index(NOTES) == {"visitors/ben": ["likes houses"],
                                "tasks/draw": ["far board"]}
  assert files.volatile()[NOTES] == files.index(NOTES)
  assert "asked for a house" not in json.dumps(files.volatile())
  # The document on the wire carries the bodies, grouped by topic.
  assert files.read(NOTES) == ("## visitors/ben\n- likes houses: asked for a house "
                               "on the far board twice\n\n## tasks/draw\n"
                               "- far board: fails from the explore's end")
  with pytest.raises(ThoughtRefused) as e:
    files.note({"topic": "tasks/draw", "title": "far board", "text": "again"}, t=3.0)
  assert "unnote it first" in str(e.value)
  for bad in ({"topic": "", "title": "x", "text": "y"}, {"topic": "a", "title": "", "text": "y"},
              {"topic": "a", "title": "x", "text": ""}, "just a string"):
    with pytest.raises(ThoughtRefused):
      files.note(bad, t=4.0)
  # `findings/...` is the science record's, written by `record` only.
  with pytest.raises(ThoughtRefused) as e:
    files.note({"topic": "findings/mass", "title": "block", "text": "0.4 kg"}, t=5.0)
  assert "record" in str(e.value)
  assert files.writes[NOTES] == 2


def test_unnote_takes_a_key_a_title_or_a_quote_and_refuses_ambiguity(files):
  files.note({"topic": "bays", "title": "bay C", "text": "it sticks"}, t=1.0)
  files.note({"topic": "bays", "title": "bay D", "text": "it is fine"}, t=2.0)
  assert files.unnote("bays/bay C", t=3.0) == "bays/bay C: it sticks"
  assert files.index(NOTES) == {"bays": ["bay D"]}
  with pytest.raises(ThoughtRefused):
    files.unnote("nothing like this", t=4.0)
  files.note({"topic": "bays", "title": "bay E", "text": "it is fine too"}, t=5.0)
  with pytest.raises(ThoughtRefused) as e:
    files.unnote("is fine", t=6.0)                   # two notes contain it
  assert "2 notes" in str(e.value)
  assert files.unnote("bay E", t=7.0).startswith("bays/bay E")
  # What was taken out is retired, findable, and never in the index again.
  assert files.records.count("pluggybot", "note", status="retired") == 2
  assert len(files.records.find("pluggybot", "sticks")) == 1


def test_the_notes_are_capped_by_count_and_refuse_out_loud(files):
  from pluggybot.mind.text import MAX_NOTES
  for i in range(MAX_NOTES):
    files.note({"topic": "fruit", "title": f"fruit {i}", "text": "tasty"}, t=1.0)
  with pytest.raises(ThoughtRefused) as e:
    files.note({"topic": "fruit", "title": "one more", "text": "tasty"}, t=2.0)
  assert "full" in str(e.value) and "unnote" in str(e.value)
  assert len(files.index(NOTES)["fruit"]) == MAX_NOTES
  assert files.unnote("fruit/fruit 0", t=3.0)
  assert files.note({"topic": "fruit", "title": "one more", "text": "tasty"}, t=4.0)


def test_findings_are_typed_topics_per_task(files):
  """`Findings.md` is the first TYPED topic family (issue #221):
  `findings/<task>`, fields declared by code, read by code (#227), shown
  by index -- `quantity = value unit`, the method on recall -- and grouped
  per task on the wire."""
  assert files.record({"quantity": "block mass", "value": 0.42, "unit": "kg",
                       "method": "the lift", "topic": "mass bench"}, t=1.0) \
    == "block mass = 0.42 kg -- the lift"
  files.record({"quantity": "plants", "value": 7, "unit": "", "method": "counted"}, t=2.0)
  assert files.findings() == [
    {"quantity": "block mass", "value": 0.42, "unit": "kg", "method": "the lift",
     "topic": "findings/mass_bench", "t": 1.0},
    {"quantity": "plants", "value": 7.0, "unit": "", "method": "counted",
     "topic": "findings/general", "t": 2.0},
  ]
  assert files.index(FINDINGS) == {"findings/mass_bench": ["block mass = 0.42 kg"],
                                   "findings/general": ["plants = 7"]}
  assert files.read(FINDINGS) == ("## findings/mass_bench\nblock mass = 0.42 kg -- "
                                  "the lift\n\n## findings/general\nplants = 7 -- counted")
  # A finding is never a note, whatever the index says.
  assert files.index(NOTES) == {}
  assert files.retract("block mass", t=3.0) == "block mass = 0.42 kg -- the lift"
  assert list(files.index(FINDINGS)) == ["findings/general"]
  with pytest.raises(ThoughtRefused):
    files.record({"quantity": "opinion", "value": "no number"}, t=4.0)


def test_a_true_death_archives_the_records_and_starts_a_generation(tmp_path):
  """What a true death costs (issue #136), on the store's terms: the rows
  stay on the volume, the next robot sees none of them, the constitution
  survives, and the views are kept beside the fresh ones."""
  root = tmp_path / "thoughts"
  files = ThoughtFiles(root)
  files.pin("bay C sticks", t=1.0)
  files.intend("tidy bay C", t=2.0)
  files.note({"topic": "bays", "title": "bay C", "text": "sticks"}, t=3.0)
  files.remember("woke up", t=4.0)
  gone = files.archive(t=5.0)
  assert gone["records"] == 4 and set(gone["cleared"]) == set(NAMES) - {MAIN}
  for name in set(NAMES) - {MAIN}:
    assert files.read(name) == "" and files.index(NOTES) == {}
  assert files.records.find("pluggybot", "bay C") == []
  assert files.records.generation("pluggybot") == 2
  assert (root / "Top_of_mind.1.md").read_text() == "bay C sticks\n"
  assert files.stats()["records"] == {"active": 0, "retired": 0, "generation": 2}
  # ...and the next process wakes up as the second robot, with nothing.
  assert ThoughtFiles(root).read(GOALS) == ""
  assert ThoughtFiles(root).records.generation("pluggybot") == 2


def test_a_pin_or_a_note_can_cite_the_history_it_came_from(files):
  """Reflection grounding (the paper's §4.3, issue #221): a lesson names
  the episodes it was drawn from, by the ids the History tail shows."""
  a = files.remember("draw on whiteboard_b failed: no route", t=1.0)
  b = files.remember("draw on whiteboard_b failed again", t=2.0)
  ids = [int(line.split()[0][1:]) for line in files.history_tail()]
  assert len(ids) == 2 and a and b
  files.pin("the far board is not worth the trip", t=3.0, cites=[f"#{ids[0]}", ids[1]])
  files.note({"topic": "boards", "title": "b", "text": "unreachable"}, t=4.0,
             cites=["#nonsense", ids[1]])
  pinned = files.records.active("pluggybot", "core", topic=TOP_OF_MIND)[0]
  noted = files.records.active("pluggybot", "note")[0]
  assert pinned.cites == tuple(ids) and noted.cites == (ids[1],)
