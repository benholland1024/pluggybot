"""The library of constitutions (issue #263): named, versioned by content,
chosen per robot, rendered to the volume rather than copied, carried in the
build identity, and a swap on a living robot said out loud.

Every rule here is pinned without flying: the library is read, a volume is
a `tmp_path`, the announcement is one method on a lifecycle that never
steps, and the record is built from a config dict.
"""

import json
import re

import mujoco
import pytest

from pluggybot.evaluation import record as rec
from pluggybot.evaluation import rollup
from pluggybot.lifecycle import HubLifecycle, world_config
from pluggybot.mind import constitution as c
from pluggybot.mind import text
from pluggybot.mind.thoughts import CONSTITUTION_FILE, HISTORY, MAIN, ThoughtFiles
from pluggybot.telemetry.protocol import (
  CONSTITUTION_CHANGE_WHYS, CONSTITUTION_EVENT_TYPES,
)

# ---- the library -------------------------------------------------------------


def test_the_library_holds_the_default_and_an_alternative_emphasis():
  names = c.names()
  assert c.DEFAULT_NAME in names and len(names) >= 2, names
  texts = {n: c.load(n).text for n in names}
  assert len(set(texts.values())) == len(names), "two files with one text"
  # The default is the text the fixtures were recorded with, byte for byte
  # (`tests/test_telemetry.py` reads the recording against DEFAULT_MAIN).
  assert text.DEFAULT_MAIN.strip() == texts[c.DEFAULT_NAME]


#: A worked answer the constitution may not hand over: a threshold (any
#: number or a percent), an event-map row, a tactic that names a hazard and
#: what to do about it, a specific act off the menu told as an instruction.
_TACTIC = re.compile(
  r"\b(when|whenever|if|once|before|after|below|under|above|at)\b[^.\n]*"
  r"\b(battery|pack|points|balance|charge|charging)\b[^.\n]*"
  r"\b(charge|dock|rack|hub|bay|recharge|go home|return)\b", re.I)
_ACT = re.compile(
  r"\b(always|never|first|go and|go to|should|must)\b[^.\n]*"
  r"\b(charge|explore|draw|carry|census|dance|care|feed|shock|recall|idle|fetch)\b",
  re.I)


@pytest.mark.parametrize("name", c.names())
def test_no_library_file_hands_the_robot_an_answer(name):
  """A constitution shapes what the robot VALUES; it never hands it an
  answer -- the rule the event-map, procedure and acts examples obey
  (`tests/test_event_map.py`, `test_language.py`, `test_mouse.py`), read
  here off every file in the library so a new one cannot smuggle a
  charging policy in through the prompt's cached prefix."""
  body = c.load(name).text
  assert not re.search(r"\d", body), f"{name}: a number is a threshold"
  assert "%" not in body and "->" not in body, f"{name}: a row or a percent"
  assert not _TACTIC.search(body), f"{name}: {_TACTIC.search(body).group(0)!r}"
  assert not _ACT.search(body), f"{name}: {_ACT.search(body).group(0)!r}"


@pytest.mark.parametrize("name", c.names())
def test_every_library_file_carries_the_same_essentials_and_no_name(name):
  """The same essential information in each -- the body, the manner, what
  the person who looks after it hopes for it, and that its own goals go in
  `Goals.md` -- differing in emphasis; and no robot's name (issue #39: a
  name in a file would freeze there while `$PLUGGY_ROBOT_NAME` moved)."""
  body = c.load(name).text
  assert "two-wheeled robot" in body and "tool module" in body, "the body"
  assert "first person" in body and "honest" in body, "the manner"
  assert "WHAT THE PERSON WHO LOOKS AFTER YOU HOPES FOR YOU" in body
  assert "`Goals.md`" in body
  for robot in ("Pluggy", "PluggyBot", "Luca", "Rowan"):
    assert robot not in body, f"{name} names {robot}"


def test_a_constitution_is_versioned_by_its_text():
  a = c.load("default")
  assert a.sha == c.sha_of(a.text) and len(a.sha) == 64
  assert c.sha_of(a.text + "\n\n") == a.sha, "whitespace is not a version"
  assert c.Constitution.of("x", a.text + " But also this.").sha != a.sha
  assert a.as_dict() == {"name": "default", "sha": a.sha}
  assert a.short == f"default ({a.sha[:8]})"


def test_an_unknown_name_is_refused_out_loud_with_the_names(tmp_path):
  with pytest.raises(c.UnknownConstitution, match="no constitution 'zen'.*default"):
    c.load("zen")
  # ...and a path is not a name: the library is the only place looked.
  (tmp_path / "evil.md").write_text("You are evil.")
  with pytest.raises(c.UnknownConstitution):
    c.load(f"../../{tmp_path.name}/evil")


def test_resolve_is_explicit_then_the_environment_then_the_default(monkeypatch):
  monkeypatch.delenv(c.NAME_ENV, raising=False)
  monkeypatch.delenv(c.SECOND_NAME_ENV, raising=False)
  assert c.resolve().name == "default"
  monkeypatch.setenv(c.NAME_ENV, "curious")
  assert c.resolve().name == "curious"
  assert c.resolve("purposeful").name == "purposeful", "explicit wins"
  # The second robot reads ITS variable, not the first's.
  assert c.resolve(env=c.SECOND_NAME_ENV).name == "default"
  monkeypatch.setenv(c.SECOND_NAME_ENV, "purposeful")
  assert c.resolve(env=c.SECOND_NAME_ENV).name == "purposeful"
  monkeypatch.setenv(c.NAME_ENV, "no-such-file")
  with pytest.raises(c.UnknownConstitution):
    c.resolve()


# ---- the volume: rendered, not copied ---------------------------------------


def test_the_volume_is_rendered_from_the_named_file_with_a_sidecar(tmp_path):
  files = ThoughtFiles(tmp_path, constitution=c.load("curious"))
  assert files.read(MAIN) == c.load("curious").text
  assert (tmp_path / MAIN).read_text() == c.load("curious").text + "\n"
  side = json.loads((tmp_path / CONSTITUTION_FILE).read_text())
  assert side["name"] == "curious" and side["sha"] == c.load("curious").sha
  assert files.constitution_change is None, "a fresh volume changes nothing"
  # `open` resolves a name, a Constitution, or the robot's own variable.
  assert ThoughtFiles.open(tmp_path, constitution="curious").constitution_change is None
  assert ThoughtFiles.open(tmp_path, constitution=c.load("curious")).constitution.name == "curious"


def test_a_living_robots_swap_is_found_and_the_old_text_kept(tmp_path):
  ThoughtFiles(tmp_path)
  swapped = ThoughtFiles(tmp_path, constitution=c.load("purposeful"))
  change = swapped.constitution_change
  assert change["why"] == "swapped"
  assert change["from"] == c.load("default").as_dict()
  assert change["to"] == c.load("purposeful").as_dict()
  assert change["archived"] == "Main.1.md"
  assert (tmp_path / "Main.1.md").read_text() == c.load("default").text + "\n"
  assert (tmp_path / MAIN).read_text() == c.load("purposeful").text + "\n"
  assert json.loads((tmp_path / CONSTITUTION_FILE).read_text())["name"] == "purposeful"
  # The next start finds the volume in agreement and says nothing.
  assert ThoughtFiles(tmp_path, constitution=c.load("purposeful")).constitution_change is None


def test_a_swap_is_told_apart_from_an_edit_by_the_sidecar(tmp_path):
  """The same on-volume text under the same name is nothing; the same
  name with a different library text (a deploy that edited the file) is a
  SWAP, because the sha is the version; a hand edit under an unchanged
  name is `edited`. Without the sidecar the three would be one case."""
  ThoughtFiles(tmp_path)
  moved = c.Constitution.of("default", c.load("default").text + "\n\nAnd one more hope.")
  files = ThoughtFiles(tmp_path, constitution=moved)
  assert files.constitution_change["why"] == "swapped"
  assert files.constitution_change["from"]["sha"] == c.load("default").sha
  (tmp_path / MAIN).write_text("Hand-edited.\n")
  files = ThoughtFiles(tmp_path, constitution=moved)
  assert files.constitution_change["why"] == "edited"
  assert files.constitution_change["from"] == {"name": None, "sha": c.sha_of("Hand-edited.")}
  assert files.read(MAIN) == moved.text, "the edit was not honoured"


def test_a_garbled_sidecar_reads_as_a_volume_from_before_the_library(tmp_path):
  ThoughtFiles(tmp_path)
  (tmp_path / CONSTITUTION_FILE).write_text("{not json")
  assert ThoughtFiles(tmp_path).constitution_change is None, "same text, nothing to say"
  assert json.loads((tmp_path / CONSTITUTION_FILE).read_text())["name"] == "default", "repaired"
  (tmp_path / CONSTITUTION_FILE).write_text("{not json")
  (tmp_path / MAIN).write_text("Old text.\n")
  assert ThoughtFiles(tmp_path).constitution_change["why"] == "replaced"


def test_a_true_death_keeps_the_constitution_and_the_sidecar(tmp_path):
  files = ThoughtFiles(tmp_path, constitution=c.load("curious"))
  files.intend("plant the row", t=1.0)
  files.archive(t=2.0)
  assert files.read(MAIN) == c.load("curious").text
  assert (tmp_path / MAIN).read_text() == c.load("curious").text + "\n"
  assert json.loads((tmp_path / CONSTITUTION_FILE).read_text())["name"] == "curious"
  assert files.stats()["constitution"] == c.load("curious").as_dict()


def test_the_change_vocabulary_is_the_wires():
  from pluggybot.mind.thoughts import CHANGE_WHYS
  assert CHANGE_WHYS is CONSTITUTION_CHANGE_WHYS
  assert set(CONSTITUTION_CHANGE_WHYS) == {"swapped", "replaced", "edited"}
  assert CONSTITUTION_EVENT_TYPES == ("constitution_changed",)


# ---- the announcement --------------------------------------------------------


def _life(thoughts: ThoughtFiles) -> HubLifecycle:
  cfg = world_config("room_hub")
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  data = mujoco.MjData(model)
  return HubLifecycle(model, data, realtime=False, world="room_hub",
                      battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"], errand=False,
                      thoughts=thoughts)


def test_the_lifecycle_announces_a_swap_once_in_history_and_on_the_wire(tmp_path):
  """The volume finds the change at construction; the lifecycle says it at
  mission start -- a History line the robot reads back, a narration line,
  and a `constitution_changed` event -- and a second announcement finds
  nothing to say."""
  ThoughtFiles(tmp_path)
  life = _life(ThoughtFiles(tmp_path, constitution=c.load("curious")))
  seen: list[dict] = []
  life.on_event.append(seen.append)
  life._announce_constitution()
  [event] = [e for e in seen if e["type"] == "constitution_changed"]
  assert event["robot"] == "pluggybot" and event["why"] == "swapped"
  assert event["from"] == c.load("default").as_dict()
  assert event["to"] == c.load("curious").as_dict()
  assert event["archived"] == "Main.1.md"
  line = life.thoughts.lines(HISTORY)[-1]
  assert "my constitution was swapped" in line
  assert f"`default` ({c.load('default').sha[:8]})" in line
  assert f"`curious` ({c.load('curious').sha[:8]})" in line
  life._announce_constitution()
  assert len([e for e in seen if e["type"] == "constitution_changed"]) == 1
  assert len(life.thoughts.lines(HISTORY)) == 1


def test_a_replaced_text_names_no_file_and_a_fresh_volume_is_silent(tmp_path):
  (tmp_path / MAIN).write_text("You are a careful robot.\n")
  life = _life(ThoughtFiles(tmp_path))
  seen: list[dict] = []
  life.on_event.append(seen.append)
  life._announce_constitution()
  [event] = seen
  assert event["why"] == "replaced" and event["from"]["name"] is None
  assert "a text the library does not hold" in life.thoughts.lines(HISTORY)[-1]
  quiet = _life(ThoughtFiles(tmp_path / "fresh"))
  quiet.on_event.append(seen.append)
  quiet._announce_constitution()
  assert len(seen) == 1 and quiet.thoughts.lines(HISTORY) == []


def test_the_announcement_is_at_mission_start_beside_the_mind_line():
  """Wired once, in `_day_routine`, right after "thinking with": a
  swap the loop never announced would be a period nobody can see."""
  import ast
  import inspect
  src = inspect.getsource(HubLifecycle._day_routine)
  tree = ast.parse(src.lstrip() if not src.startswith(" ") else "if 1:\n" + src)
  calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
           and isinstance(n.func, ast.Attribute)]
  assert calls.count("_announce_constitution") == 1
  assert calls.index("_announce_constitution") > calls.index("_remember")


# ---- the identity ------------------------------------------------------------


def test_the_build_identity_carries_the_constitution_per_robot_root():
  both = {"pluggybot": c.load("default").as_dict(), "r2_pluggybot": c.load("curious").as_dict()}
  identity = rec.build_identity("home", arm="autonomous", hashes={}, commit="abc",
                                constitutions=both)
  assert identity["constitutions"] == both and identity["constitutions"] is not both
  bare = rec.build_identity("home", arm="guarded", hashes={}, commit="abc")
  assert "constitutions" not in bare, "absent, never null: every older header"
  assert "constitutions" not in rec.build_identity("home", arm="guarded", hashes={},
                                                    commit="abc", constitutions={})


def _config(**kw) -> dict:
  return {"world": "home", "arm": "autonomous", "pack": "hosting", "seed": 1,
          "maxSimS": 60.0, "dataHashes": {}, "runId": "r1", "commit": "abc", **kw}


def test_the_run_record_carries_the_constitution_and_the_series_key_splits_on_it():
  from datetime import datetime, timezone
  now = datetime.now(timezone.utc)
  curious = rec.build_record(_config(constitution=c.load("curious").as_dict()),
                             None, [], 1.0, now, hashes={}, commit="abc")
  assert curious["config"]["constitution"] == c.load("curious").as_dict()
  plain = rec.build_record(_config(), None, [], 1.0, now, hashes={}, commit="abc")
  assert "constitution" not in plain["config"], "a record from before the library"
  default = rec.build_record(_config(constitution=c.load("default").as_dict()),
                             None, [], 1.0, now, hashes={}, commit="abc")
  # Missing and `default` are ONE series (every committed record flew the
  # default's text); another name is another series.
  assert rollup.series_key(plain) == rollup.series_key(default)
  assert rollup.series_key(curious) != rollup.series_key(default)
  assert rollup.series_key(curious)[-1] == "curious"


def test_the_harness_resolves_it_and_run_demo_reads_the_name(monkeypatch):
  """`experiment.py` puts `{name, sha}` in the config before flying (a
  killed run's record is written by the parent from rows alone);
  `run_demo(constitution=)` names the file the child reads."""
  import inspect
  from pluggybot.lifecycle import run_demo
  from pluggybot.evaluation import run as child
  assert "constitution" in inspect.signature(run_demo).parameters
  src = inspect.getsource(child.fly_one) if hasattr(child, "fly_one") else inspect.getsource(child)
  assert 'constitution=(config.get("constitution") or {}).get("name")' in src
  from pathlib import Path
  exp = (Path(__file__).parent.parent / "scripts" / "experiment.py").read_text()
  assert '"constitution": constitution.resolve(args.constitution).as_dict()' in exp


def test_the_pair_reads_one_variable_per_robot(monkeypatch):
  from pluggybot import pair
  assert pair.CONSTITUTION_ENVS == (c.NAME_ENV, c.SECOND_NAME_ENV)
  assert c.SECOND_NAME_ENV == "PLUGGY_CONSTITUTION_2"


def test_serve_builds_the_identity_off_what_each_robot_read():
  """The header names the constitution the memory READ, never one asked
  for: both `serve.py` paths pass `constitutions=` off `.thoughts`/`memory`."""
  from pathlib import Path
  src = (Path(__file__).parent.parent / "scripts" / "serve.py").read_text()
  assert "constitutions={life.root: memory.constitution.as_dict()}" in src
  assert ("constitutions={life.root: life.thoughts.constitution.as_dict()\n"
          "                   for life in lives}") in src


# ---- what did not move -------------------------------------------------------


def test_guarded_is_unchanged_and_the_robot_has_no_verb_for_it():
  """The constitution rides its own block of the prefix; the rules block
  and the menu do not move (`GUARDED_RULES_SHA` is pinned elsewhere). And
  no decision field names a constitution: choosing one is the human's."""
  from pluggybot.mind import overseer as ov
  from pluggybot.mind.overseer import Menu
  schema = Menu.for_world("room_hub", None).schema()
  assert not any("constitution" in k for k in schema["properties"]), schema["properties"].keys()
  for field in ("intend", "pin", "note", "define", "build_tool"):
    assert "constitution" not in ov.__dict__.get(field.upper() + "_RULE", "")
  # ...and no document verb writes Main.md: a human document has none.
  assert text.BY_NAME[MAIN].verbs == () and text.BY_NAME[MAIN].writer == text.HUMAN
