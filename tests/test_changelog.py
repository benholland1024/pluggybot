"""The changelog (issue #617): what the people who run the world wrote for
the robots about how it changed, read by them through `recall`, its newest
date in every turn's context beside today's, and on the wire once a stream
where a mind reads it.

Every rule pinned without flying: the file is parsed, its entries are read
for what they may not carry (each rule shown to catch a sentence that breaks
it), `recall` runs on the ThoughtFiles alone, the context on a stub life, the
prefix twice, and the open burst on the recorder's mini world.
"""

import queue
import re
import threading
from pathlib import Path

import mujoco
import pytest

from pluggybot.lifecycle import overseer_context
from pluggybot.mind import changelog
from pluggybot.mind import overseer as ov
from pluggybot.mind.changelog import Changelog, Malformed
from pluggybot.mind.overseer import Menu, Overseer
from pluggybot.mind.thoughts import RECALLED_CHARS, ThoughtFiles
from pluggybot.telemetry.recorder import FrameBuilder

from test_body import stub_life
from test_constitution import _ACT, _TACTIC
from test_humanoid_words import _foreign
from test_overseer import FakeClient
from test_telemetry import MINI_XML, _MappedMind, record

REPO = Path(__file__).resolve().parents[1]
SKILL = REPO / ".claude" / "skills" / "pw-release" / "SKILL.md"


@pytest.fixture()
def mini_model():
  return mujoco.MjModel.from_xml_string(MINI_XML)


# ---- the file ------------------------------------------------------------------


def test_the_committed_changelog_is_newest_first_each_date_once():
  log = changelog.current()
  assert log.entries, "the changelog has no entry"
  dates = [e.date for e in log.entries]
  assert dates == sorted(dates, reverse=True) and len(set(dates)) == len(dates)
  assert log.newest == dates[0]
  for entry in log.entries:
    assert entry.items, entry.date
    for item in entry.items:
      assert len(item) <= changelog.MAX_ITEM_CHARS, f"{entry.date}: {item[:60]!r}"
  assert changelog.PATH.parent == Path(changelog.__file__).parent, \
    "the file ships beside the module, as the constitutions do"


def test_the_file_is_one_shape_and_anything_else_is_refused_with_its_line():
  ok = Changelog.parse("# Title\n\n<!-- a note\nfor people -->\n\n## 2026-11-07\n\n"
                       "- One fact, written\n  over two lines.\n- Another.\n\n"
                       "## 2026-11-06\n\n- An older one.\n")
  assert [e.as_dict() for e in ok.entries] == [
    {"date": "2026-11-07", "items": ["One fact, written over two lines.", "Another."]},
    {"date": "2026-11-06", "items": ["An older one."]}]
  assert ok.lines() == ["[2026-11-07] One fact, written over two lines.",
                        "[2026-11-07] Another.", "[2026-11-06] An older one."]
  # The version is the entries as the robots read them: a comment or a
  # re-wrapped line is not a new one.
  same = Changelog.parse("## 2026-11-07\n- One fact, written over two lines.\n"
                         "- Another.\n## 2026-11-06\n- An older one.\n")
  assert same.sha == ok.sha and len(ok.sha) == 64
  for text, why in (("## 2026-11-06\n- a\n## 2026-11-07\n- b\n", "not older"),
                    ("## 2026-11-07\n- a\n## 2026-11-07\n- b\n", "not older"),
                    ("## 2026-02-30\n- a\n", "not a date"),
                    ("## 2026-11-07\n\n## 2026-11-06\n- a\n", "nothing in it"),
                    ("## 2026-11-07\nA fact with no dash.\n", "line 2"),
                    ("- an item before any date\n", "line 1"),
                    ("## 2026-11-07\n- a\n# A title after an entry\n", "line 3")):
    with pytest.raises(Malformed, match=why):
      Changelog.parse(text)


# ---- what an entry may not carry (issue #617, Ben's rules) ---------------------


#: OUR SIDE OF IT: what we measure or why, how anything is graded, what we
#: hope they will do. ⚠ "study room" is a place's name; why it is there is
#: never said.
MEASURING = re.compile(
  r"\b(measur\w*|metrics?|scor(e|es|ed|ing)|grad(e|es|ed|er|ers|ing)|observator\w*|"
  r"observations?|evaluat\w*|experiment\w*|periods?|qualit(y|ies)|hypothes\w*|"
  r"hop(e|es|ed|ing)|expect\w*|stud(y(?! room)|ies|ied)|subjects?|"
  r"research(?! center)\w*|baselines?|ablations?|rungs?|benchmark\w*|"
  r"we (want|think|believe|wonder|plan)\w*|to (see|find out|learn|test) (if|whether|how))\b",
  re.I)
#: A HIDDEN TRUTH: the mouse's nature (`overseer.DISCLOSURE` is the one line
#: ever said of it), a prop's hidden draws, the held-out tasks. An entry
#: about the mouse at all is Ben's call, made by changing this line.
HIDDEN = re.compile(
  r"\b(mouse|mice|beyond (this|the) simulation|connected to anything|held[- ]out|"
  r"secrets?|hidden|truth|true (position|pose|state|nature)|draws?|drawn|dynamics|"
  r"stiff\w*|damp\w*|friction|seeds?|seeded|random\w*)\b", re.I)
#: ADVICE: it says what changed, never what to do about it.
ADVICE = re.compile(
  r"\b(you (should|must|might want|may want|could try|need to|ought|had better)|"
  r"it is (best|wise|better|a good idea)|it'?s (best|wise|better)|"
  r"we (suggest|recommend|advise)|try to|make sure|be sure|remember to|"
  r"don'?t forget|worth (trying|doing)|so that you|in order to|"
  r"consider\w*|prefer\w*|avoid\w*|instead of)\b", re.I)
#: ...and a sentence that opens on a verb is an instruction.
IMPERATIVE = re.compile(
  r"(?:^|[.!?:;]\s+)(Try|Use|Go|Walk|Look|Charge|Plug|Explore|Recall|Read|Write|Define|"
  r"Keep|Avoid|Remember|Check|Plan|Wait|Ask|Take|Claim|Make|Set|Stay|Don'?t|Do|Never|"
  r"Always|Consider|Prefer|Be|Note|Pin|Choose|Pick|Head|Find|Stop|Start|Leave|Give|"
  r"Tell|Say|See)\b")
#: ...and nothing that points at our side of the world.
OURS = re.compile(r"#\d|\b\w+\.(py|md|json|ts|vue|npz|xml)\b|\b(PRs?|pull requests?|"
                  r"commits?|branch(es)?|github|repo(sitory)?|staging)\b", re.I)
NAMES = ("Pluggy", "PluggyBot", "Luca", "Rowan")


def _prose(item: str) -> str:
  """An item as the rules read it: a code span names a verb, a key or a
  field the robot is told of elsewhere (`measure`, `heldTasks`), so its
  words are not the entry's."""
  return re.sub(r"`[^`]*`", "`x`", item)


def faults(item: str) -> list[str]:
  """Every rule `item` breaks, by name -- none for an entry that may ship."""
  prose = _prose(item)
  out = [name for name, rule in (("measuring", MEASURING), ("hidden", HIDDEN),
                                 ("advice", ADVICE), ("imperative", IMPERATIVE),
                                 ("tactic", _TACTIC), ("act", _ACT), ("ours", OURS))
         if rule.search(prose)]
  if "%" in item or "->" in item:
    out.append("a threshold or a row")
  if any(name in item for name in NAMES):
    out.append("a robot's name")
  if _foreign(prose):
    out.append("another body's world")
  if "<!--" in item:
    out.append("a comment inside an item")
  return out


@pytest.mark.parametrize("entry", changelog.current().entries, ids=lambda e: e.date)
def test_no_entry_carries_what_the_robots_may_not_read(entry):
  """The rules (Ben, 2026-11-06; docs/Overseer.md §7b): facts about their
  world and their body, never what we measure or why, how anything is
  graded, a hidden truth or what we hope they will do, and no advice -- on
  the rule every prompt's rule obeys (`test_constitution.py`'s tactics and
  acts, `test_humanoid_words.py`'s other bodies), read off every item."""
  for item in entry.items:
    assert not faults(item), (entry.date, faults(item), item)


#: One sentence for each rule, each of which must be caught: a rule that
#: cannot fail is decor.
BAD = (
  ("measuring", "We measure how often each of you goes out to the park."),
  ("measuring", "Every hand job is graded by how far the drawer slid out."),
  ("measuring", "We hope you will both explore the mall soon."),
  ("measuring", "The study room is for studies with the robots as subjects."),
  ("measuring", "Your looks are read in the observatory now."),
  ("hidden", "The mouse may be connected to anything beyond this simulation."),
  ("hidden", "Each drawer's stiffness is drawn for its own job."),
  ("hidden", "The held-out chests begin at a million."),
  ("advice", "It is best to read this after every restart."),
  ("advice", "You should keep away from the park's west sidewalk."),
  ("imperative", "There are three stations. Use the mall's when the house's is taken."),
  ("tactic", "When your battery is low, go to the station and charge."),
  ("act", "You must always charge first."),
  ("a threshold or a row", "A pack under 30 % is a long walk from the mall."),
  ("a threshold or a row", "nothing_to_do -> ask is a row worth keeping."),
  ("a robot's name", "Luca's room is the west one."),
  ("another body's world", "The rack is gone from the workshop."),
  ("ours", "See #560 for the waist, merged to staging."),
)


@pytest.mark.parametrize("rule,sentence", BAD, ids=[r for r, _ in BAD])
def test_each_rule_catches_a_sentence_that_breaks_it(rule, sentence):
  assert rule in faults(sentence), (rule, faults(sentence))


# ---- recall reads it, a page at a time ------------------------------------------


def _long(n_entries=3, items=5, chars=380) -> Changelog:
  """A changelog longer than one recall block, its items each its own."""
  text = ""
  for e in range(n_entries):
    text += f"## 2026-11-{20 - e:02d}\n"
    for i in range(items):
      text += f"- Entry {e} fact {i}: " + "word " * ((chars - 20) // 5) + "\n"
  return Changelog.parse(text)


def test_read_changelog_is_its_newest_page_and_says_how_many_there_are(monkeypatch):
  """`changelog` (issue #617) is a key like `chat`, ahead of the notes: its
  newest page, whole items, each line its entry's date first, and `page` /
  `pages`; `changelog 2` the next, and a page past the last has no lines.
  Shown to fail by reading `changelog` as a note topic: no hits."""
  log = _long()
  monkeypatch.setattr(changelog, "current", lambda: log)
  files = ThoughtFiles()
  first = files.recall(read="changelog")
  assert first["page"] == 1 and first["pages"] == len(log.pages(RECALLED_CHARS)) == 2
  assert first["lines"][0].startswith("[2026-11-20] Entry 0 fact 0")
  assert first["hits"] == len(first["lines"]) and "cut" not in first
  assert sum(len(line) + 1 for line in first["lines"]) <= RECALLED_CHARS + 1
  second = files.recall(read="changelog 2")
  assert second["page"] == 2 and second["lines"]
  assert first["lines"] + second["lines"] == log.lines(), "no item lost between pages"
  assert files.recall(read="Changelog  page 2")["lines"] == second["lines"]
  past = files.recall(read="changelog 3")
  assert past["lines"] == [] and past["hits"] == 0 and past["pages"] == 2


def test_a_note_topic_called_changelog_is_read_by_its_title_and_find_is_memorys():
  files = ThoughtFiles()
  files.note({"topic": "changelog", "title": "mall", "text": "the mall is new"}, t=1.0)
  files.remember("walked to the park", t=2.0)
  assert files.recall(read="changelog/mall")["lines"] == [
    "#1 [changelog/mall] the mall is new"]
  both = files.recall(read="changelog", find="park")
  assert both["page"] == 1 and both["lines"][:-1] == changelog.current().page(1, RECALLED_CHARS)[0]
  assert both["lines"][-1].startswith("#2 [history]"), "find searches memory, not the changelog"


# ---- every turn carries its date, and the prefix never does ----------------------


def _mind():
  boss = Overseer(Menu.for_world("home_humanoid", None), client=FakeClient())
  return boss, stub_life(overseer=boss)


def test_every_turn_carries_the_newest_date_beside_today(monkeypatch):
  """Ben, 2026-11-06: the newest entry's date in the PER-TURN state, with
  a date of the robot's own beside it -- the context had none -- and the
  rule naming both."""
  monkeypatch.setattr(changelog, "today", lambda now=None: "2026-11-09")
  _, life = _mind()
  state = overseer_context(life)
  assert state["changelog"] == {"newest": changelog.current().newest, "today": "2026-11-09"}
  shown = ov.model_state(dict(state))
  assert shown["changelog"] == state["changelog"], "the view keeps it"
  assert '"changelog"' in ov._user_turn(shown)
  # The rule tells it what the block and the key are, and prescribes
  # nothing about either.
  assert "`changelog` for the changelog" in ov.RULES
  for key in state["changelog"]:
    assert f"(`{key}`)" in ov.RULES, key
  assert "`changelog 2` is the next page" in ov.RULES


def test_today_is_the_utc_date():
  from datetime import datetime, timedelta, timezone
  late = datetime(2026, 11, 7, 23, 30, tzinfo=timezone(timedelta(hours=-3)))
  assert changelog.today(late) == "2026-11-08", "the entries are dated in UTC"


def test_a_new_entry_moves_no_prompt_and_shows_in_the_turn(monkeypatch):
  """A release adds an entry; the cached prefix must not move with it, or
  every release is a new prompt sha and a cold cache (#617: per-turn,
  never the prefix). Shown to fail by putting the newest date in a rule."""
  before, life = _mind()
  newer = Changelog.parse("## 2099-01-01\n- A newer fact.\n")
  monkeypatch.setattr(changelog, "current", lambda: newer)
  after = Overseer(Menu.for_world("home_humanoid", None), client=FakeClient())
  assert after.prompt_sha == before.prompt_sha
  prefix = "".join(p["text"] for p in after.system)
  assert "2099-01-01" not in prefix and "A newer fact" not in prefix
  assert overseer_context(life)["changelog"]["newest"] == "2099-01-01"


# ---- the wire: once a stream, where a mind reads it ------------------------------


def test_a_stream_with_a_mind_opens_with_the_changelog_once(mini_model, tmp_path):
  log = changelog.current()
  _, lines = record(mini_model, seconds=0.4, tmp=tmp_path, thoughts=ThoughtFiles(),
                    overseer=_MappedMind([]))
  first_frame = next(i for i, x in enumerate(lines) if "type" not in x)
  opened = [x for x in lines[:first_frame] if x.get("type") == "changelog"]
  assert len(opened) == 1 and not [x for x in lines[first_frame:]
                                   if x.get("type") == "changelog"]
  (msg,) = opened
  assert msg["sha"] == log.sha and msg["newest"] == log.newest and "robot" not in msg
  assert msg["entries"] == [e.as_dict() for e in log.entries]
  # A world no mind reads sends none: not an empty changelog.
  _, plain = record(mini_model, seconds=0.4, tmp=tmp_path, thoughts=ThoughtFiles())
  assert not [x for x in plain if x.get("type") == "changelog"]


def test_a_live_consumer_is_handed_it_on_every_connect(mini_model):
  from pluggybot.telemetry.publisher import WsPublisher

  data = mujoco.MjData(mini_model)
  pub = WsPublisher.__new__(WsPublisher)          # no socket, no sender thread
  pub._builder = FrameBuilder(mini_model, data, model_name="mini",
                              thoughts=ThoughtFiles(), overseer=_MappedMind([]))
  pub.data = data
  pub._queue = queue.Queue(maxsize=64)
  for flag in ("_need_goals", "_need_thoughts", "_need_boards", "_need_keyframe"):
    setattr(pub, flag, threading.Event())
  pub.boards, pub._grids = None, []
  pub.frames_dropped = pub.events_dropped = pub.events_queued = pub.events_sent = 0
  for _ in range(2):                              # a connect, and another
    pub._need_thoughts.set()
    pub.step_hook()
  sent = [m for kind, m in list(pub._queue.queue) if kind == "event"]
  assert [m["sha"] for m in sent if m["type"] == "changelog"] == [changelog.current().sha] * 2


# ---- the release skill drafts it -------------------------------------------------


def test_the_release_skill_names_the_file_and_the_rules_it_drafts_under():
  """The skill (`/pw-release`) drafts each entry under these rules and
  stops for Ben; a path it names that moved would leave it drafting into
  nothing, so each must exist (the `when=` fence's rule)."""
  text = SKILL.read_text()
  head = re.match(r"---\n(.*?)\n---\n", text, re.S)
  assert head and re.search(r"^name: pw-release$", head.group(1), re.M)
  assert re.search(r"^description: \S", head.group(1), re.M)
  for path in re.findall(r"`((?:src|tests|docs)/[\w./-]+)`", text):
    assert (REPO / path).exists(), path
  for named in ("src/pluggybot/mind/changelog.md", "tests/test_changelog.py"):
    assert f"`{named}`" in text, named
