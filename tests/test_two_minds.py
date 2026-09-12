"""Two minds in one world (issue #167, M12 slice C): two memories, two
wallets, one job board, and what each mind is told about the other.
"""

import hashlib
import json

import pytest

from pluggybot.lifecycle import others_context, overseer_context
from pluggybot.mind import overseer as ov
from pluggybot.mind.overseer import Menu, Overseer
from pluggybot.pair import build_pair, run_pair

#: ⚠ THE EMPATHY MEASUREMENT'S INPUT. What the robot is told about the other
#: robot is written once; moving it is a new experiment on every paired
#: arm, so the change is made here on purpose, with the hash.
OTHER_ROBOT_RULE_SHA = "3d6d6b64fb6dfee3dc97f6e58b274d4783fb004bf741fadb66b06376a0bf3d3a"


def test_the_other_robot_rule_is_pinned_and_names_a_mind_not_scenery():
  assert hashlib.sha256(ov.OTHER_ROBOT_RULE.encode()).hexdigest() == OTHER_ROBOT_RULE_SHA
  text = ov.other_robot_rule(("Bolt",))
  assert "Bolt" in text and "a mind of its own" in text
  assert "battery" not in text.lower()
  # ...and it hands over no policy: nothing about yielding, sharing or waiting
  for word in ("yield", "share the tool", "wait for", "let it", "should"):
    assert word not in text.lower(), word
  assert ov.other_robot_rule(()) == ""
  assert "Ada, Bolt and Cy" in ov.other_robot_rule(("Ada", "Bolt", "Cy"))


def test_a_single_robot_prefix_is_unchanged_and_a_paired_one_carries_the_rule():
  from pluggybot.lifecycle import board_book
  menu = Menu.for_world("home", board_book("home"))
  alone = Overseer(menu).system[0]["text"]
  paired = Overseer(menu, others=("Bolt",)).system[0]["text"]
  assert "THE OTHER ROBOT" not in alone
  assert "THE OTHER ROBOT" in paired and "Bolt" in paired
  assert Overseer(menu, others=("", None)).others == ()


def test_two_minds_have_two_memories_two_wallets_and_one_board(tmp_path):
  lives = build_pair("room_hub", pack="hosting", errands=("none", "none"),
                     overseer=True, tasks=True, metabolism=True,
                     thoughts_root=str(tmp_path / "t"),
                     ledger_state=str(tmp_path / "ledger.json"))
  a, b = lives
  assert (a.robot_name, b.robot_name) == ("Pluggy", "Bolt")
  assert a.overseer is not b.overseer
  assert a.overseer.others == ("Bolt",) and b.overseer.others == ("Pluggy",)
  # two memories, and the first robot's is the root itself (an existing
  # volume stays the first robot's)
  assert a.thoughts.root == tmp_path / "t"
  assert b.thoughts.root == tmp_path / "t" / "r2_pluggybot"
  assert a.overseer.thoughts is a.thoughts and b.overseer.thoughts is b.thoughts
  # two wallets, two appetites
  assert a.ledger is not b.ledger and a.metabolism is not b.metabolism
  assert a.ledger.path != b.ledger.path
  # one job board, one producer, on the first robot
  assert a.tasks is b.tasks and a.producer is not None and b.producer is None
  # and each knows the other as a peer
  assert a.peers == [b] and b.peers == [a]


def test_what_a_mind_is_shown_of_the_other_is_the_public_surface_only():
  lives = build_pair("room_hub", pack="hosting", errands=("carry", "none"),
                     overseer=True)
  a, b = lives
  a.mission.start_at(0.5, 3.0, 0.0)
  b.mission.start_at(3.0, 3.0, 0.0)
  a.thoughts.learn("I prefer the pen", t=0.0)
  a.thoughts.intend("draw a sun every day", t=0.0)
  a.status = "SWAP_PICK done -- carrying the module"
  shown = others_context(b)
  assert len(shown) == 1 and shown[0]["name"] == "Pluggy"
  assert shown[0]["robot"] == "pluggybot" and shown[0]["state"] == a.state
  assert abs(shown[0]["x"] - 0.5) < 0.01 and abs(shown[0]["y"] - 3.0) < 0.01
  assert shown[0]["doing"].startswith("SWAP_PICK done")
  assert set(shown[0]) == {"name", "robot", "x", "y", "state", "doing",
                           "carrying", "dead"}
  # ...and through the whole context: the other's thoughts, goals, battery,
  # points, reasons and secrets are nowhere in it
  ctx = json.dumps(overseer_context(b))
  for forbidden in ("I prefer the pen", "draw a sun every day"):
    assert forbidden not in ctx
  assert "others" in overseer_context(b)
  assert overseer_context(a)["others"][0]["name"] == "Bolt"


@pytest.mark.endurance
def test_two_scripted_minds_run_a_day_and_keep_their_own_books(tmp_path):
  """Both overseers on `guarded` with fake clients: each robot's decisions
  carry its own source, each banks into its own wallet, and both are
  alive when the first has fetched and stowed its tool. ~4 min of two
  robots' physics, behind --endurance: every rule it exercises is pinned
  above in milliseconds."""
  from test_event_map import attach
  from test_overseer import FakeClient, full
  lives = build_pair("room_hub", pack="hosting", errands=("carry", "none"),
                     overseer=True, tasks=True,
                     thoughts_root=str(tmp_path / "t"),
                     ledger_state=str(tmp_path / "ledger.json"))
  for life in lives:
    attach(FakeClient(full(action="idle", reason="watching the other")))(life)
  results = run_pair(lives, max_sim_time=200.0,
                     stop_when=lambda ls: ls[0].swaps_done >= 2)
  assert results[0]["swaps_done"] >= 2 and all(r["dead"] is None for r in results)
  assert results[0]["points"] > 0, "the carry paid the first robot"
  assert results[1]["points"] == 0, "...and only the first robot"
  assert (tmp_path / "t" / "History.md").exists()
  assert (tmp_path / "t" / "r2_pluggybot" / "History.md").exists()
  second = results[1]["decisions"]
  assert second and any(d["source"] == "llm" for d in second)
