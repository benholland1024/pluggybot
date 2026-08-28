"""The thought files: named memory with per-file write permissions (#38).

Four documents, four different answers to "who may write this", and the whole
module is the enforcement of that table plus the two places the enforcement
could be quietly wrong.

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

import pytest

from pluggybot.hub import overseer as ov
from pluggybot.hub.lifecycle import board_book, world_config
from pluggybot.hub.overseer import Menu, Overseer
from pluggybot.hub.thoughts import (
  GOALS, HISTORY, HUMAN, KNOWLEDGE, MAIN, MAX_LINE_CHARS, ROBOT, SYSTEM,
  NAMES, SPECS, ThoughtFiles, ThoughtRefused,
)
from pluggybot.telemetry.protocol import THOUGHT_FILES, THOUGHT_WRITERS

from test_overseer import FakeClient, full


@pytest.fixture()
def files(tmp_path):
  return ThoughtFiles(tmp_path / "thoughts")


# ---- the permission table ----------------------------------------------------


def test_the_four_files_and_their_writers_are_the_wire_vocabulary():
  """One list, two repos. A rename here that missed protocol.py would put a
  document on the wire under a name no client has a renderer for."""
  assert NAMES == THOUGHT_FILES
  assert {s.writer for s in SPECS.values()} <= set(THOUGHT_WRITERS)
  assert SPECS[MAIN].writer == SPECS[GOALS].writer == HUMAN
  assert SPECS[HISTORY].writer == SYSTEM
  assert SPECS[KNOWLEDGE].writer == ROBOT


@pytest.mark.parametrize("name,by", [
  (MAIN, ROBOT), (MAIN, SYSTEM),          # nothing writes the persona
  (GOALS, ROBOT), (GOALS, SYSTEM),        # nor the goals
  (HISTORY, ROBOT),                       # the robot cannot edit its past
  (KNOWLEDGE, SYSTEM),                    # ...and code does not think for it
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


def test_the_robot_writes_exactly_one_file(files):
  """...and the one it writes is its own."""
  assert files.learn("whiteboard_b is the one people look at", t=2.0)
  assert files.read(KNOWLEDGE) == "whiteboard_b is the one people look at"
  # The other three are unreachable from the robot's two verbs: `learn` and
  # `unlearn` name KNOWLEDGE in code, and the general `append` checks the
  # table. There is no verb that takes a file name from a model at all.
  assert files.writes[KNOWLEDGE] == 1
  assert all(files.writes[n] == 0 for n in NAMES if n != KNOWLEDGE)


def test_a_human_file_is_read_from_disk_and_never_written_back(tmp_path):
  root = tmp_path / "thoughts"
  ThoughtFiles(root)
  # Missing human files are materialised once, so there is something to edit.
  assert (root / MAIN).exists() and (root / GOALS).exists()
  (root / GOALS).write_text("Draw a robot on every wall.\n")

  reopened = ThoughtFiles(root)
  assert reopened.read(GOALS) == "Draw a robot on every wall."
  # A whole run's worth of the robot's own writing must not touch it.
  reopened.learn("the garden is bigger than it looks", t=3.0)
  reopened.record("charged to 92%", t=4.0)
  assert (root / GOALS).read_text() == "Draw a robot on every wall.\n"


def test_the_pre_38_goals_file_still_wins(tmp_path):
  """A deploy has been editing `/var/lib/pluggybot/goals.md` since #15.
  Pointing it at a directory must not silently revert that to the defaults
  -- $PLUGGY_GOALS keeps meaning "this file is Goals.md"."""
  legacy = tmp_path / "goals.md"
  legacy.write_text("Water the garden, then rest.\n")
  files = ThoughtFiles(tmp_path / "thoughts", goals_path=legacy)
  assert files.read(GOALS) == "Water the garden, then rest."
  assert not (tmp_path / "thoughts" / GOALS).exists(), \
    "a second Goals.md was written beside the one being read"
  assert legacy.read_text() == "Water the garden, then rest.\n"


def test_a_fresh_deploy_finds_files_to_edit(tmp_path):
  """The human half is only editable if a person can FIND it. A missing
  Main.md or Goals.md is written out with the defaults the robot is already
  living by -- at the resolved path, so `$PLUGGY_GOALS` gets the file rather
  than the thoughts directory getting a second one."""
  root, legacy = tmp_path / "thoughts", tmp_path / "goals.md"
  files = ThoughtFiles(root, goals_path=legacy)
  assert (root / MAIN).read_text().strip() == files.read(MAIN)
  assert legacy.read_text().strip() == files.read(GOALS)
  # The files code and the robot write are NOT created up front: they do not
  # exist until there is something in them, and a bootstrap is not a write.
  assert not (root / HISTORY).exists() and not (root / KNOWLEDGE).exists()
  assert all(n == 0 for n in files.writes.values())

  # ...and reading without a root creates nothing at all: `goals_text(path)`
  # must not have a file as a side effect.
  bare = tmp_path / "elsewhere.md"
  ThoughtFiles(goals_path=bare)
  assert not bare.exists()


# ---- the caps ----------------------------------------------------------------


def test_history_rolls_and_knowledge_refuses(files):
  """Both are size caps and they fail in OPPOSITE directions, deliberately.

  History is a rolling record nothing curates, so the oldest line falls off
  -- the journal's rule. Knowledge is the robot's own, and dropping the
  oldest line silently would mean a robot that believes it remembers
  something it does not; it is refused instead, loudly, and `forget` is the
  robot's remedy.
  """
  for i in range(400):
    files.record(f"something happened, number {i}", t=float(i))
  assert len(files.read(HISTORY)) <= SPECS[HISTORY].cap
  assert files.dropped[HISTORY] > 0
  assert "number 399" in files.read(HISTORY)
  assert "number 0" not in files.read(HISTORY), "an unbounded history"

  with pytest.raises(ThoughtRefused) as e:
    for i in range(400):
      files.learn(f"an opinion about board number {i}", t=float(i))
  assert "full" in str(e.value)
  assert len(files.read(KNOWLEDGE)) <= SPECS[KNOWLEDGE].cap
  assert files.dropped[KNOWLEDGE] == 0, \
    "the robot's own file dropped a line it was never told about"


def test_a_long_line_is_trimmed_rather_than_refused(files):
  """A cap on ONE line is not the same failure as a full file: prose cut
  short is still prose, so it is trimmed at the write path -- the journal
  note's rule -- while a file with no room left is refused."""
  written = files.learn("x" * (MAX_LINE_CHARS * 3), t=1.0)
  assert len(written) == MAX_LINE_CHARS
  assert files.learn("", t=1.0) == "", "an empty line is cost with no content"
  assert files.writes[KNOWLEDGE] == 1


def test_forget_is_how_the_robot_changes_its_mind(files):
  files.learn("whiteboard_a is the one people look at", t=1.0)
  files.learn("bay C sticks a little", t=2.0)
  assert files.unlearn("whiteboard_a is the one people look at", t=3.0)
  assert files.lines(KNOWLEDGE) == ["bay C sticks a little"]
  # A quote that picks out exactly one line works; the point is that the
  # robot does not have to reproduce its own sentence to the character.
  assert files.unlearn("bay C sticks", t=4.0)
  assert files.lines(KNOWLEDGE) == []


@pytest.mark.parametrize("quote,why", [
  ("a thing it never wrote", "nothing"),
  ("board", "2 lines"),
])
def test_a_forget_that_does_not_pick_out_one_line_is_refused(files, quote, why):
  """Ambiguity and a miss are both refusals, because the alternative is a
  robot that asked to drop one belief and dropped another -- or believes it
  dropped one and did not."""
  files.learn("board a is nearly full", t=1.0)
  files.learn("board b is empty", t=2.0)
  with pytest.raises(ThoughtRefused) as e:
    files.unlearn(quote, t=3.0)
  assert why in str(e.value)
  assert len(files.lines(KNOWLEDGE)) == 2


def test_there_is_no_verb_that_rewrites_a_file(files):
  """"Append or patch, not blind rewrite" -- a full-rewrite verb lets one
  bad generation erase everything the robot knows. The public surface is
  two verbs and neither can empty the page in one call."""
  assert not hasattr(files, "write")
  assert not hasattr(files, "replace")
  verbs = {"append", "forget", "learn", "unlearn", "record"}
  assert verbs <= set(dir(files))


# ---- persistence -------------------------------------------------------------


def test_the_files_survive_a_restart(tmp_path):
  """World state, on the terms the boards and the ledger are: every mission
  end is a restart, so a memory that did not survive one is not memory."""
  root = tmp_path / "thoughts"
  first = ThoughtFiles(root)
  first.learn("the far whiteboard is not worth the trip", t=10.0)
  first.record("drew a house on whiteboard_a", t=20.0)

  second = ThoughtFiles(root)
  assert second.read(KNOWLEDGE) == "the far whiteboard is not worth the trip"
  assert "drew a house on whiteboard_a" in second.read(HISTORY)
  # ...and a second day appends to the first day's record rather than
  # starting a fresh one.
  second.record("woke up in home", t=0.0)
  assert len(ThoughtFiles(root).lines(HISTORY)) == 2


def test_an_in_memory_set_writes_nothing(tmp_path):
  """Every unit test, spike and demo without a state directory. It must be
  usable and must not litter the working directory."""
  files = ThoughtFiles()
  files.learn("nothing to see", t=1.0)
  assert files.read(KNOWLEDGE) == "nothing to see"
  assert not list(tmp_path.iterdir())


# ---- the wire ----------------------------------------------------------------


def test_a_document_goes_out_whole_and_says_who_writes_it(files):
  files.learn("the garden is bigger than it looks", t=5.0)
  msg = files.message(KNOWLEDGE, t=5.0)
  assert msg["type"] == "thought" and msg["name"] == KNOWLEDGE
  assert msg["writer"] == ROBOT and msg["cap"] == SPECS[KNOWLEDGE].cap
  assert msg["text"] == files.read(KNOWLEDGE)
  assert msg["robot"] == "pluggybot"
  # JSON-serialisable, like every other typed message.
  assert json.loads(json.dumps(msg)) == msg
  # All four open a stream, in reading order.
  assert [m["name"] for m in files.messages(t=5.0)] == list(NAMES)


def test_every_change_is_published_as_it_happens(files):
  """The site cannot read these files -- it runs on a different box -- so a
  change that is not streamed is a change the Thoughts tab never shows."""
  seen = []
  files.on_event.append(seen.append)
  files.learn("bay C sticks", t=1.0)
  files.record("charged to 92%", t=2.0)
  assert [m["name"] for m in seen] == [KNOWLEDGE, HISTORY]
  assert seen[0]["text"] == "bay C sticks"
  # A REFUSED write publishes nothing: the file did not change.
  with pytest.raises(ThoughtRefused):
    files.append(GOALS, "mine now", by=ROBOT, t=3.0)
  assert len(seen) == 2


# ---- the prompt cache, which is the issue's trap ------------------------------


def test_only_the_human_files_ride_the_cached_prefix():
  """The split, from both sides. `volatile` is derived from the same flag
  `stable` reads, so the halves cannot disagree -- and the way two
  hand-written lists WOULD disagree is a file that reaches the model
  through neither, which looks from outside exactly like a robot that never
  learns anything."""
  files = ThoughtFiles()
  assert set(files.stable()) == {MAIN, GOALS}
  assert set(files.volatile()) == {HISTORY, KNOWLEDGE}
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
  files.learn("bay C sticks a little", t=1.0)
  files.record("charged to 92%", t=2.0)
  turn = ov._user_turn({"thoughts": files.volatile()})
  assert "bay C sticks a little" in turn
  assert "charged to 92%" in turn


def test_the_prefix_does_not_move_when_the_robot_writes():
  """The issue's second acceptance criterion, and the reason the split is
  by WRITER rather than by content: the two files that change during a run
  are exactly the two the robot and the code can write."""
  files = ThoughtFiles()
  menu = Menu(boards=("whiteboard_a",), programs=("house",))
  boss = Overseer(menu, thoughts=files, client=FakeClient())
  before = boss.system[0]["text"]

  files.learn("whiteboard_b is the one people look at", t=1.0)
  files.record("drew a house on whiteboard_a", t=2.0)

  assert boss.system[0]["text"] == before, \
    "a self-edit moved the cached prefix -- every call now pays full price"
  assert boss.system[0]["cache_control"] == {"type": "ephemeral"}
  # ...and the writable files are genuinely reaching the model, in the turn
  # AFTER the breakpoint. Absent from both halves would pass the assertion
  # above for the worst possible reason.
  #
  # ⚠ Their CONTENT, not their names: the rules block in the prefix talks
  # ABOUT `Knowledge_and_Opinions.md`, which is stable text and exactly
  # right. The same distinction `test_the_stable_prefix_is_byte_identical`
  # draws between quoted JSON keys and prose.
  assert "whiteboard_b is the one people look at" not in before
  assert "drew a house on whiteboard_a" not in before
  turn = ov._user_turn({"thoughts": files.volatile()})
  assert "whiteboard_b is the one people look at" in turn
  assert "drew a house on whiteboard_a" in turn


def test_the_history_the_model_sees_is_the_tail(files):
  """The whole file is on disk and on the wire; the PROMPT gets the last few
  lines. The last dozen things that happened are context, and the hundred
  before them are input tokens on every call for the rest of the mission."""
  from pluggybot.hub.thoughts import HISTORY_SHOWN
  for i in range(HISTORY_SHOWN * 3):
    files.record(f"thing number {i}", t=float(i))
  shown = files.volatile()[HISTORY]
  assert len(shown) == HISTORY_SHOWN
  assert "thing number 0" not in " ".join(shown)
  assert f"thing number {HISTORY_SHOWN * 3 - 1}" in " ".join(shown)
  # ...and nothing was lost from the file itself to achieve that.
  assert len(files.lines(HISTORY)) > HISTORY_SHOWN


def test_a_human_edit_is_the_one_thing_that_should_move_it(tmp_path):
  """The other side of the same coin: editing Goals.md between runs SHOULD
  invalidate the cache, because the prefix genuinely changed. This is not a
  bug being tolerated -- it is the reason the split is safe."""
  root = tmp_path / "thoughts"
  menu = Menu(boards=("whiteboard_a",), programs=("house",))
  first = Overseer(menu, thoughts=ThoughtFiles(root), client=FakeClient())
  (root / GOALS).write_text("Draw a robot on every wall.\n")
  second = Overseer(menu, thoughts=ThoughtFiles(root), client=FakeClient())
  assert first.system != second.system
  assert "Draw a robot on every wall." in second.system[0]["text"]


def test_the_persona_is_the_file_rather_than_the_code(tmp_path):
  """Main.md is what makes 'who the robot is' editable without a redeploy,
  which is the whole reason it is a file. If the prompt kept a hardcoded
  persona alongside it, editing the file would change nothing visible."""
  root = tmp_path / "thoughts"
  ThoughtFiles(root)                       # materialise the defaults
  (root / MAIN).write_text("You are a very serious robot. Never dawdle.\n")
  boss = Overseer(Menu(boards=("a",), programs=("house",)),
                  thoughts=ThoughtFiles(root), client=FakeClient())
  text = boss.system[0]["text"]
  assert "You are a very serious robot." in text
  assert "two-wheeled robot" not in text, \
    "the default persona is still in the prompt beside the edited one"


# ---- the robot's own writes, through a real decision --------------------------


def test_a_decision_can_learn_and_forget_without_spending_a_turn():
  """`learn`/`forget` are orthogonal to the action, like `note` and
  `respond_to`: a robot that had to spend its turn to write a line down
  writes fewer of them than it should."""
  menu = Menu(boards=("whiteboard_a",), programs=("house",))
  boss = Overseer(menu, client=FakeClient(
    full(action="draw", board="whiteboard_a", program="house",
         learn="people look at whiteboard_a more than b")))
  decision = boss.decide({})
  assert decision.action == "draw"          # the action still happened
  assert decision.learn == "people look at whiteboard_a more than b"
  assert decision.source == "llm"


def test_the_schema_offers_both_verbs_and_no_third(menu_home):
  """A model can add a line and remove a line. There is deliberately no
  parameter that names a FILE, so `Main.md` is not reachable from a decision
  at all -- the permission check is a backstop, not the only lock."""
  props = menu_home.schema()["properties"]
  assert "learn" in props and "forget" in props
  assert not [k for k in props if "file" in k.lower()]
  assert "learn" in menu_home.schema()["required"]


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
                                         learn="bay C sticks a little")))
  life = _lifecycle("room_hub", overseer=boss, thoughts=files, errand=False)
  said: list[str] = []
  life.say_hooks.append(lambda t, line: said.append(line))
  life.mission.start_at(*world_config("room_hub")["start"])
  try:
    life._decide()
  finally:
    life.mission.close()

  assert "chose carry: tidying up" in files.read(HISTORY)
  assert files.read(KNOWLEDGE) == "bay C sticks a little"
  assert any(line.startswith("THOUGHT learn: bay C sticks") for line in said)
  # ...and it is on disk, because the next mission is a different process.
  assert "bay C sticks a little" in (tmp_path / "thoughts"
                                     / KNOWLEDGE).read_text()


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
      files.learn(f"an opinion about board number {i}", t=float(i))

  life._reconsider(ov.Decision(action="idle", learn="one thing too many"))
  assert any("THOUGHT refused" in line and "full" in line for line in said)
  assert "one thing too many" not in files.read(KNOWLEDGE)
  # ...and a mission is never ended by a memory write.
  assert life.thoughts.refusals
