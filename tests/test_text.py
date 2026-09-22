"""One protocol for text (issue #217): every surface is a document or a
message, behind one registry, with storage behind one interface.

What is pinned here, cheaply -- no physics, no network:

  THE TABLE IS THE SOURCE. Each module that owns a surface reads its writer,
  its cap and its verbs off its row; a cap typed in two places is how the
  prompt tells the robot one number and the gate enforces another.

  ONE GATE. `text.admit` answers "may this writer make this document this
  big" for every document, and the three answers (yes / roll / refuse) are
  the row's policy. A fourth document gets its answer by filling in a row.

  ONE PATH TO THE DISK. Nothing that owns a document writes a file itself;
  `mind/store.py` does, and the syntax trees say so. That is what makes the
  memory mechanism a single change when it is rethought (#221).

  TWO SHAPES, AND NOTHING CROSSES. A message has a sender and is never a
  document; a document has a writer and is never sent. The security
  argument (a message never framed like the robot's own file) and the
  one-writer rule both depend on it.

  THE SCIENCE RECORD is a row: `record` writes one measured finding in a
  shape code reads back, `retract` takes one off, and `guarded` sees none
  of it -- the verbs ride the library's slot, the document is hidden from
  a menu that does not offer them, and the prefix is byte-identical.
"""

import ast
import hashlib
import re
from pathlib import Path

import pytest

from pluggybot.mind import inbox, overseer as ov, store, text, thoughts
from pluggybot.mind import tickets as desk
from pluggybot.mind.overseer import Menu, Overseer
from pluggybot.mind.thoughts import FINDINGS, GOALS, HISTORY, MAIN, NOTES, TOP_OF_MIND, ThoughtFiles
from pluggybot.procedure import library as procedures
from pluggybot.telemetry.protocol import THOUGHT_FILES, THOUGHT_VERBS, THOUGHT_WRITERS
from pluggybot.workshop import library as workshop

from test_autonomous import GUARDED_RULES_SHA
from test_overseer import FakeClient, full

SRC = Path(__file__).resolve().parent.parent / "src" / "pluggybot"

#: The modules that OWN a surface. Every one reads its rules off the
#: registry and its bytes off the store; none touches the disk itself.
OWNERS = ("mind/thoughts.py", "procedure/library.py", "workshop/library.py",
          "mind/inbox.py", "mind/text.py", "mind/tickets.py")
#: What a disk write looks like in a syntax tree: a method only a Path has
#: (a string has none of these), a bare `open`, or an `os` / `shutil` call
#: that moves bytes. `str.replace` is not `os.replace`.
PATH_METHODS = {"write_text", "write_bytes", "unlink", "mkdir", "rmdir", "touch"}
OS_CALLS = {"replace", "rename", "remove", "unlink", "mkdir", "makedirs",
            "rmdir", "move", "copy", "rmtree"}


def _disk_writes(path: Path) -> set[str]:
  """Every call in a module that could write the disk."""
  tree = ast.parse(path.read_text())
  out = set()
  for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
      continue
    f = node.func
    if isinstance(f, ast.Name) and f.id == "open":
      out.add("open")
    elif isinstance(f, ast.Attribute):
      owner = getattr(f.value, "id", "")
      if f.attr in PATH_METHODS or (owner in ("os", "shutil") and f.attr in OS_CALLS):
        out.add(f"{owner + '.' if owner else ''}{f.attr}")
  return out


def _calls(path: Path) -> set[str]:
  """Every attribute or name called in a module."""
  tree = ast.parse(path.read_text())
  out = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Call):
      f = node.func
      out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
  return out


# ---- the table is the source ------------------------------------------------


def test_every_text_surface_is_a_registry_row():
  """Six files, two libraries, the ticket desk, four message channels:
  thirteen rows, and each owner's numbers are the row's. A cap or a writer
  typed in an owner would be a second table."""
  names = {s.name for s in text.SURFACES}
  assert names == {*THOUGHT_FILES, "procedures", "tools", "tickets",
                   "visitor", "peer", "library", "operator"}
  assert thoughts.SPECS == {s.name: s for s in text.FILES}
  assert thoughts.NAMES == THOUGHT_FILES
  assert procedures.Library(None).cap == text.BY_NAME["procedures"].cap == text.MAX_PROCEDURES
  assert len(workshop.BAYS) == text.BY_NAME["tools"].cap
  assert procedures.SUFFIX == text.BY_NAME["procedures"].suffix
  assert workshop.SUFFIX == text.BY_NAME["tools"].suffix
  assert inbox.MAX_TEXT == text.BY_NAME["visitor"].cap == text.BY_NAME["peer"].cap
  assert ov.MAX_TELL == text.BY_NAME["peer"].cap
  # ...and the desk (#284): the cap counts OPEN tickets and is the row's.
  assert desk.MAX_OPEN == text.BY_NAME["tickets"].cap == text.MAX_OPEN_TICKETS
  assert desk.SUFFIX == text.BY_NAME["tickets"].suffix
  # ⚠ A LINE OF A THREAD IS A TICKET'S TEXT, NOT A MESSAGE'S (the length
  # follow-up on #284): one number for both directions, off the operator
  # row, and the queue's own cap is a visitor's sentence and stays one.
  assert desk.MAX_LINE == text.BY_NAME["operator"].cap == text.MAX_TICKET_CHARS
  assert desk.MAX_TEXT == text.MAX_TICKET_CHARS > inbox.MAX_TEXT
  assert inbox.MAX_TICKET_TEXT == text.MAX_TICKET_CHARS
  # ...and the library's page is a paragraph, not a sentence (#216): its
  # own cap, read by the module that delivers it.
  from pluggybot.mind import wiki
  assert wiki.MAX_PAGE_CHARS == text.BY_NAME["library"].cap > inbox.MAX_TEXT
  # ...and the verbs on the `.md` files are the wire's, plus the refusal.
  assert set(THOUGHT_VERBS) == {v for s in text.FILES for v in s.verbs} | {"refused"}
  assert text.line_verbs() == ("drop_goal", "intend", "unpin", "pin",
                               "retract", "record", "unnote", "note")


def test_the_two_shapes_stay_two():
  """A document has a writer off THOUGHT_WRITERS and a policy; a message
  has a sender and no policy but the queue's. No row is both, and the
  gate refuses a message outright: a sender must never be able to write."""
  for s in text.DOCUMENTS:
    assert s.shape == text.DOCUMENT and s.writer in THOUGHT_WRITERS
    assert s.policy in text.POLICIES
    assert (s.policy == text.NONE) == (s.writer == text.HUMAN)
    assert s.stable == (s.writer == text.HUMAN), s.name
    assert len(s.verbs) <= 2, f"{s.name}: an add, a remove, and no third"
    assert (s.verbs == ()) == (s.writer != text.ROBOT), s.name
  for m in text.MESSAGES:
    assert m.shape == text.MESSAGE and m.writer in (text.VISITOR, text.PEER,
                                                    text.LIBRARY, text.OPERATOR)
    assert m.writer not in THOUGHT_WRITERS or m.writer == "robot"
    assert not m.stable and not m.default and not m.offered_with
    with pytest.raises(text.Refused, match="is a message"):
      text.admit(m, m.writer, 1)
  # A robot's message and a robot's document share the word `robot` and
  # nothing else: the peer row has no verbs that write anything.
  assert text.BY_NAME["peer"].verbs == text.BY_NAME["visitor"].verbs
  assert not set(text.BY_NAME["peer"].verbs) & set(text.BY_VERB)


# ---- one gate ------------------------------------------------------------------


@pytest.mark.parametrize("row", [s for s in text.DOCUMENTS], ids=lambda s: s.name)
def test_the_gate_answers_every_document_row(row):
  """Three answers, one per policy, and the wrong writer gets none of them."""
  others = [w for w in THOUGHT_WRITERS if w != row.writer]
  for by in others:
    with pytest.raises(text.Refused, match=row.name):
      text.admit(row, by, 0)
  if row.policy == text.NONE:
    with pytest.raises(text.Refused, match="person editing the file"):
      text.admit(row, row.writer, 0)
    return
  assert text.admit(row, row.writer, row.cap) is True
  if row.policy == text.ROLL:
    assert text.admit(row, row.writer, row.cap + 1) is False
  else:
    with pytest.raises(text.Refused, match=f"{row.name} is full"):
      text.admit(row, row.writer, row.cap + 1)
  # A narrower cap for one instance is honoured; a wider one is not.
  with pytest.raises(text.Refused) if row.policy == text.REFUSE else pytest.raises(AssertionError):
    assert text.admit(row, row.writer, 2, cap=1)
  assert text.admit(row, row.writer, row.cap, cap=row.cap * 10) is True


def test_every_owner_passes_the_gate_and_none_touches_the_disk():
  """The grep, as a syntax-tree walk: the three document owners call
  `admit`, and no owner calls anything that writes a file -- only the
  store does. A bespoke write path is exactly a module that fails this."""
  for rel in ("mind/thoughts.py", "procedure/library.py", "workshop/library.py",
              "mind/tickets.py"):
    assert "admit" in _calls(SRC / rel), f"{rel} writes without the gate"
  for rel in OWNERS:
    hit = _disk_writes(SRC / rel)
    assert not hit, f"{rel} writes the disk itself: {sorted(hit)}"
  assert _disk_writes(SRC / "mind/store.py") >= {"tmp.write_text", "os.replace"}


def test_a_full_library_and_a_full_workshop_refuse_through_the_gate():
  """The two entry documents at their caps, in memory: the refusal names
  the document and the remedy, and the record is untouched."""
  lib = procedures.Library(None, cap=1)
  lib.define("one", "def one():\n  look()\n")
  with pytest.raises(procedures.LibraryRefused, match="library is full .*undefine one first"):
    lib.define("two", "def two():\n  look()\n")
  assert lib.names() == ("one",)
  shop = workshop.Workshop(None)
  row = text.BY_NAME["tools"]
  for i in range(row.cap):
    shop.entries[f"t{i}"] = workshop.Entry(f"t{i}", {}, i, {}, 0.0, None)
  with pytest.raises(workshop.WorkshopRefused, match="workshop is full .*retire_tool one first"):
    shop.check("t9", {"name": "t9"}, "A")
  assert len(shop.entries) == row.cap


# ---- one path to the disk ---------------------------------------------------------


def test_the_two_stores_behave_alike(tmp_path):
  """`FileStore` and `MemoryStore` answer the same sequence the same way,
  so a test on memory is a test on the volume."""
  def drive(s):
    s.write("a.md", "one\n")
    s.write("sub/b.procedure", "def b():\n  look()\n")
    out = [s.read("a.md"), s.read("missing"), s.keys(), s.keys(suffix=".md")]
    out.append(s.archive("a.md"))
    out.append(s.archive("a.md"))        # nothing there any more
    s.write("a.md", "two\n")
    out.append(s.archive("a.md"))        # the next number
    s.remove("sub/b.procedure")
    s.remove("sub/b.procedure")          # removing nothing is nothing
    out.append(s.keys())
    return out
  assert drive(store.FileStore(tmp_path / "v")) == drive(store.MemoryStore()) == [
    "one\n", None, ["a.md", "sub/b.procedure"], ["a.md"],
    "a.1.md", None, "a.2.md", ["a.1.md", "a.2.md"]]
  assert isinstance(store.FileStore(None), store.Store)
  assert (tmp_path / "v" / "a.2.md").read_text() == "two\n"


def test_the_thought_files_and_both_libraries_live_on_the_store(tmp_path):
  """Written through the store, read back by a fresh owner off the same
  root -- and the file names on the volume are the ones the operator
  knows (`<Name>.md`, `<name>.procedure`, `<name>.tool.json`)."""
  root = tmp_path / "thoughts"
  files = ThoughtFiles(root)
  files.pin("bay C sticks", t=1.0)
  assert isinstance(files.store, store.FileStore)
  assert (root / TOP_OF_MIND).read_text() == "bay C sticks\n"
  assert ThoughtFiles(root).read(TOP_OF_MIND) == "bay C sticks"
  lib = procedures.Library(None, root=root / "procedures")
  lib.define("look_around", "def look_around():\n  look()\n")
  assert (root / "procedures" / "look_around.procedure").exists()
  assert procedures.Library(None, root=root / "procedures").names() == ("look_around",)
  assert isinstance(workshop.Workshop(root / "tools").store, store.FileStore)
  assert isinstance(ThoughtFiles().store, store.MemoryStore)


# ---- the science record ------------------------------------------------------------


def test_the_science_record_is_code_checkable():
  """A finding goes in as a number and comes back as one; prose is refused
  out loud; a retraction takes it off the record."""
  files = ThoughtFiles()
  line = files.record({"quantity": "mass of the unknown block", "value": 0.42,
                       "unit": "kg", "method": "lift current under load"}, t=1.0)
  assert line == "mass of the unknown block = 0.42 kg -- lift current under load"
  assert files.findings() == [{"quantity": "mass of the unknown block",
                               "value": 0.42, "unit": "kg",
                               "method": "lift current under load",
                               "topic": "findings/general", "t": 1.0}]
  assert files.record({"quantity": "count", "value": "7"}) == "count = 7"
  assert files.findings()[-1] == {"quantity": "count", "value": 7.0, "unit": "",
                                  "method": "", "topic": "findings/general", "t": 0.0}
  for bad in ({"quantity": "mass", "value": "about a kilo"},
              {"quantity": "", "value": 1}, "the block is heavy",
              {"quantity": "x", "value": float("nan")}):
    with pytest.raises(thoughts.ThoughtRefused, match="a finding is a quantity"):
      files.record(bad)
  assert files.retract("count = 7") == "count = 7"
  assert [f["quantity"] for f in files.findings()] == ["mass of the unknown block"]
  # A line the robot could only write in another shape is not a finding.
  assert thoughts.parse_finding("the block is heavy") is None
  assert thoughts.parse_finding("g = -9.81 m/s^2 -- a drop") == {
    "quantity": "g", "value": -9.81, "unit": "m/s^2", "method": "a drop"}
  assert files.spec(FINDINGS).writer == text.ROBOT
  assert files.spec(FINDINGS).policy == text.REFUSE


def test_the_record_is_the_robots_and_goes_with_a_true_death(tmp_path):
  files = ThoughtFiles(tmp_path / "t")
  files.record({"quantity": "q", "value": 1}, t=1.0)
  with pytest.raises(thoughts.ThoughtRefused):
    files.append(FINDINGS, "q = 2", by=text.SYSTEM)
  gone = files.archive()
  assert FINDINGS in gone["cleared"] and MAIN not in gone["cleared"]
  assert files.read(FINDINGS) == "" and (tmp_path / "t" / "Findings.1.md").exists()


def test_the_record_is_offered_with_the_library_and_hidden_from_guarded():
  """The verbs ride the library's slot, the document rides the context
  only where they do, the rule rides the autonomous prefix only, and the
  guarded prefix is the byte-identical control."""
  assert hashlib.sha256(ov.RULES.encode()).hexdigest() == GUARDED_RULES_SHA
  assert "Findings.md" not in ov.RULES and "Findings.md" in ov.FINDINGS_RULE
  menu = Menu(boards=("whiteboard_a",), programs=("house",))
  assert "record" not in menu.schema()["properties"]
  props = menu.schema(procedures=())["properties"]
  assert props["record"]["properties"]["value"] == {"type": "number"}
  assert "retract" in props and "record" in menu.schema(procedures=())["required"]
  files = ThoughtFiles()
  assert FINDINGS not in files.volatile(menu)
  assert FINDINGS in files.volatile(ov.replace(menu, procedures=True))
  assert set(files.volatile(menu)) == {GOALS, HISTORY, TOP_OF_MIND, NOTES}
  # A guarded parse DROPS the fields; the decision stands.
  boss = Overseer(menu, client=FakeClient(
    full(action="idle", record={"quantity": "q", "value": 1, "unit": "", "method": ""},
         retract="q = 1")))
  d = boss.decide({})
  assert d.action == "idle" and d.record is None and d.retract == ""
  assert "record" not in d.as_dict()
  # ...and the rule's example shows no answer (EVENT_MAP_RULE's rule).
  for word in ("charge", "battery", "rack", "%"):
    assert word not in ov.FINDINGS_RULE.lower(), word
  # The prefix: the rule rides the autonomous text only (since #221 the
  # memory section itself is shared -- the tiers are on every arm).
  from pluggybot.economy.scoring import default_table
  def prefix(**kw):
    return ov.system_prompt(files, menu, default_table(), **kw)[0]["text"]
  assert "Top_of_mind.md" in ov.RULES and "Notes.md" in ov.RULES
  assert ov.FINDINGS_RULE in prefix(autonomous=True, procedures=True)
  assert ov.FINDINGS_RULE not in prefix() and "Findings.md" not in prefix()


def test_the_record_reaches_the_mission_through_the_one_write_path(tmp_path):
  """`_reconsider` iterates the registry's verbs: a `record` and a
  `retract` land in `Findings.md` and are narrated `THOUGHT <verb>: <line>`,
  through the same loop as `pin` -- no branch was added for them."""
  from test_overseer import _lifecycle
  files = ThoughtFiles(tmp_path / "thoughts")
  life = _lifecycle("room_hub", thoughts=files, errand=False)
  said = []
  life.say_hooks.append(lambda t, line: said.append(line))
  life._reconsider(ov.Decision(action="idle",
                               record={"quantity": "q", "value": 2.5, "unit": "m"}))
  assert files.findings() == [{"quantity": "q", "value": 2.5, "unit": "m", "method": "",
                               "topic": "findings/general", "t": files.findings()[0]["t"]}]
  life._reconsider(ov.Decision(action="idle", retract="q = 2.5"))
  life._reconsider(ov.Decision(action="idle", record={"quantity": "q", "value": "no"}))
  verbs = [re.match(r"THOUGHT ([a-z_]+):", ln).group(1) for ln in said if ln.startswith("THOUGHT")]
  assert verbs == ["record", "retract", "refused"]
  assert files.findings() == []
  # the loop reads the registry, not a list of its own
  src = (SRC / "lifecycle.py").read_text()
  body = src[src.index("def _reconsider"):src.index("def _peer")]
  assert "line_verbs()" in body and '"pin"' not in body
